    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        is_reg = any(batch.get_is_reg_list())
        additional_loss = 0.0

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    # this only works if doing a prior pred
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)

                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std

                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier

                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust

                        eps = 1e-5
                        # match the noise to the prior
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()

            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W
                        lat_height = batch.latents.shape[3]
                        lat_width = batch.latents.shape[4]
                    else: 
                        lat_height = batch.latents.shape[2]
                        lat_width = batch.latents.shape[3]
                    # resize to size of noise_pred
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
                    # stack first channel to match channels of noise_pred
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
                    
                    if len(noise_pred.shape) == 5:
                        prior_mask = prior_mask.unsqueeze(2)  # add time dimension back for video
                        prior_mask = prior_mask.repeat(1, 1, noise_pred.shape[2], 1, 1) 

                    prior_mask_multiplier = 1.0 - prior_mask
                    
                    # scale so it is a mean of 1
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if hasattr(self.sd, 'get_loss_target'):
                    target = self.sd.get_loss_target(
                        noise=noise, 
                        batch=batch, 
                        timesteps=timesteps,
                    ).detach()
                elif self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        
        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(
                noise=noise, 
                batch=batch, 
                timesteps=timesteps,
            ).detach()
            
        elif self.sd.is_flow_matching:
            # forward ODE
            target = (noise - batch.latents).detach()
            # reverse ODE
            # target = (batch.latents - noise).detach()
        else:
            target = noise
            
        if self.dfe is not None:
            if self.dfe.version == 1:
                model = self.sd
                if model is not None and hasattr(model, 'get_stepped_pred'):
                    stepped_latents = model.get_stepped_pred(noise_pred, noise)
                else:
                    # stepped_latents = noise - noise_pred
                    # first we step the scheduler from current timestep to the very end for a full denoise
                    bs = noise_pred.shape[0]
                    noise_pred_chunks = torch.chunk(noise_pred, bs)
                    timestep_chunks = torch.chunk(timesteps, bs)
                    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
                    stepped_chunks = []
                    for idx in range(bs):
                        model_output = noise_pred_chunks[idx]
                        timestep = timestep_chunks[idx]
                        self.sd.noise_scheduler._step_index = None
                        self.sd.noise_scheduler._init_step_index(timestep)
                        sample = noisy_latent_chunks[idx].to(torch.float32)
                        
                        sigma = self.sd.noise_scheduler.sigmas[self.sd.noise_scheduler.step_index]
                        sigma_next = self.sd.noise_scheduler.sigmas[-1] # use last sigma for final step
                        prev_sample = sample + (sigma_next - sigma) * model_output
                        stepped_chunks.append(prev_sample)
                    
                    stepped_latents = torch.cat(stepped_chunks, dim=0)
                    
                stepped_latents = stepped_latents.to(self.sd.vae.device, dtype=self.sd.vae.dtype)
                sl = stepped_latents
                if len(sl.shape) == 5:
                    # video B,C,T,H,W
                    sl = sl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                    b, t, c, h, w = sl.shape
                    sl = sl.reshape(b * t, c, h, w)
                pred_features = self.dfe(sl.float())
                with torch.no_grad():
                    bl = batch.latents
                    bl = bl.to(self.sd.vae.device)
                    if len(bl.shape) == 5:
                        # video B,C,T,H,W
                        bl = bl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                        b, t, c, h, w = bl.shape
                        bl = bl.reshape(b * t, c, h, w)
                    target_features = self.dfe(bl.float())
                    # scale dfe so it is weaker at higher noise levels
                    dfe_scaler = 1 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1).to(self.device_torch)
                
                dfe_loss = torch.nn.functional.mse_loss(pred_features, target_features, reduction="none") * \
                    self.train_config.diffusion_feature_extractor_weight * dfe_scaler
                additional_loss += dfe_loss.mean()
            elif self.dfe.version == 2:
                # version 2
                # do diffusion feature extraction on target
                with torch.no_grad():
                    rectified_flow_target = noise.float() - batch.latents.float()
                    target_feature_list = self.dfe(torch.cat([rectified_flow_target, noise.float()], dim=1))
                
                # do diffusion feature extraction on prediction
                pred_feature_list = self.dfe(torch.cat([noise_pred.float(), noise.float()], dim=1))
                
                dfe_loss = 0.0
                for i in range(len(target_feature_list)):
                    dfe_loss += torch.nn.functional.mse_loss(pred_feature_list[i], target_feature_list[i], reduction="mean")
                
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight * 100.0
            elif self.dfe.version in [3, 4, 5]:
                dfe_loss = self.dfe(
                    noise=noise,
                    noise_pred=noise_pred,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    batch=batch,
                    scheduler=self.sd.noise_scheduler
                )
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight 
            else:
                raise ValueError(f"Unknown diffusion feature extractor version {self.dfe.version}")
        
        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                # we make cached blank prompt embeds that match the batch size
                unconditional_embeds = concat_prompt_embeds(
                    [self.unconditional_embeds] * noisy_latents.shape[0],
                )
                unconditional_target = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )
                is_video = len(target.shape) == 5
                
                if self.train_config.do_guidance_loss_cfg_zero:
                    # zero cfg
                    # ref https://github.com/WeichenFan/CFG-Zero-star/blob/cdac25559e3f16cb95f0016c04c709ea1ab9452b/wan_pipeline.py#L557
                    batch_size = target.shape[0]
                    positive_flat = target.view(batch_size, -1)
                    negative_flat = unconditional_target.view(batch_size, -1)
                    # Calculate dot production
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    # Squared norm of uncondition
                    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
                    st_star = dot_product / squared_norm

                    alpha = st_star
                    
                    alpha = alpha.view(batch_size, 1, 1, 1) if not is_video else alpha.view(batch_size, 1, 1, 1, 1)
                else:
                    alpha = 1.0

                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)
                
                unconditional_target = unconditional_target * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)

            if self.train_config.do_differential_guidance:
                with torch.no_grad():
                    guidance_scale = self.train_config.differential_guidance_scale
                    target = noise_pred + guidance_scale * (target - noise_pred)
            
        if target is None:
            target = noise

        pred = noise_pred

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            assert not self.train_config.train_turbo

            # --------- берём sigmas из scheduler по timesteps ---------
            # timesteps: обычно shape [B] или [B, 1]
            if timesteps.dim() > 1:
                t_flat = timesteps.view(-1)
            else:
                t_flat = timesteps

            # индекс сигмы для каждого t (ДЕЛАЕМ НА CPU!)
            sigma_indices = [
                self.sd.noise_scheduler.index_for_timestep(int(t.item()))
                for t in t_flat
            ]
            # индексы на том же девайсе, что и sigmas (обычно cpu)
            sigmas_tensor = self.sd.noise_scheduler.sigmas
            sigma_indices = torch.tensor(
                sigma_indices, device=sigmas_tensor.device, dtype=torch.long
            )

            # сначала индексируем на CPU, потом переносим на девайс модели
            sigmas_flat = sigmas_tensor[sigma_indices].to(
                noise_pred.device, dtype=noise_pred.dtype
            )  # [B]

            # приведём sigmas к виду [B, 1, 1, 1(,1)] под размер noise_pred
            sigma_shape = [sigmas_flat.shape[0]] + [1] * (noise_pred.dim() - 1)
            sigmas = sigmas_flat.view(*sigma_shape)  # Bx1x1x1 или Bx1x1x1x1

            # --------- денойзим латенты: x0 ≈ xt - σ * εθ ---------
            denoised_latents = noisy_latents - sigmas * noise_pred

            # веса: σ^-2, скаляры на батч: [B]
            weighing_flat = sigmas_flat ** -2.0
            w_shape = [weighing_flat.shape[0]] + [1] * (denoised_latents.dim() - 1)
            weighing = weighing_flat.view(*w_shape)
            # --------- задаём target ---------
            if loss_target == 'source':
                # сравниваем с латентами из батча
                target = batch.latents.to(denoised_latents.device, dtype=denoised_latents.dtype)

            elif loss_target == 'unaugmented':
                with torch.no_grad():
                    unaugmented_latents = self.sd.encode_images(
                        batch.unaugmented_tensor
                    ).to(self.device_torch, dtype=dtype)
                    unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                    target = unaugmented_latents.detach()

                # приводим target к тому виду, который ожидает prediction_type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    # x0-таргет: сравниваем денойзнутые латенты с "чистыми"
                    target = target.to(denoised_latents.device, dtype=denoised_latents.dtype)
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    # тут уже таргет не x0, а скорость
                    target = self.sd.noise_scheduler.get_velocity(
                        target.to(noise.device, dtype=noise.dtype),
                        noise,
                        timesteps,
                    ).to(denoised_latents.device, dtype=denoised_latents.dtype)
                else:
                    raise ValueError(
                        f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}"
                    )

            # --------- считаем loss без редукции ---------
            diff = denoised_latents.float() - target.float()
            loss_per_element = weighing.float() * (diff ** 2)
            loss = loss_per_element
        else:

            if self.train_config.loss_type == "mae":
                loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            elif self.train_config.loss_type == "wavelet":
                loss = wavelet_loss(pred, batch.latents, noise)
            elif self.train_config.loss_type == "stepped":
                loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
                # the way this loss works, it is low, increase it to match predictable LR effects
                loss = loss * 10.0
            elif self.train_config.loss_type == "mse_l1":
                print("In mse_l1")
                loss = mse_l1_loss(pred.float(), target.float())
            else:
                print("default")
                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
                
            do_weighted_timesteps = False
            if self.sd.is_flow_matching:
                if self.train_config.linear_timesteps or self.train_config.linear_timesteps2:
                    do_weighted_timesteps = True
                if self.train_config.timestep_type == "weighted":
                    # use the noise scheduler to get the weights for the timesteps
                    do_weighted_timesteps = True

            # handle linear timesteps and only adjust the weight of the timesteps
            if do_weighted_timesteps:
                # calculate the weights for the timesteps
                timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    v2=self.train_config.linear_timesteps2,
                    timestep_type=self.train_config.timestep_type
                ).to(loss.device, dtype=loss.dtype)
                if len(loss.shape) == 4:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1).detach()
                elif len(loss.shape) == 5:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1, 1).detach()
                loss = loss * timestep_weight

        if self.train_config.do_prior_divergence and prior_pred is not None:
            loss = loss + (torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none") * -1.0)

        if self.train_config.train_turbo:
            mask_multiplier = mask_multiplier[:, 3:, :, :]
            # resize to the size of the loss
            mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(pred.shape[2], pred.shape[3]), mode='nearest')

        # multiply by our mask
        try:
            if len(noise_pred.shape) == 5:
                # video B,C,T,H,W
                mask_multiplier = mask_multiplier.unsqueeze(2)  # add time dimension back for video
                mask_multiplier = mask_multiplier.repeat(1, 1, noise_pred.shape[2], 1, 1)
            loss = loss * mask_multiplier
        except Exception as e:
            # todo handle mask with video models
            print("Could not apply mask multiplier to loss")
            print(e)
            pass

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None and prior_mask_multiplier is not None:
            assert not self.train_config.train_turbo
            if self.train_config.loss_type == "mae":
                prior_loss = torch.nn.functional.l1_loss(pred.float(), prior_pred.float(), reduction="none")
            else:
                prior_loss = torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none")

            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if torch.isnan(prior_loss).any():
                print_acc("Prior loss is nan")
                prior_loss = None
            else:
                if len(noise_pred.shape) == 5:
                    # video B,C,T,H,W
                    prior_loss = prior_loss.mean([1, 2, 3, 4])
                else:
                    prior_loss = prior_loss.mean([1, 2, 3])
                # loss = loss + prior_loss
                # loss = loss + prior_loss
            # loss = loss + prior_loss
        if len(noise_pred.shape) == 5:
            loss = loss.mean([1, 2, 3, 4])
        else:
            loss = loss.mean([1, 2, 3])
        # apply loss multiplier before prior loss
        # multiply by our mask
        try:
            loss = loss * loss_multiplier
        except:
            # todo handle mask with video models
            pass
        if prior_loss is not None:
            loss = loss + prior_loss

        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                # add snr_gamma
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                # add snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma,
                                        fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                # add min_snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        loss = loss.mean()


        if (
            not self.train_config.train_turbo
            and self.train_config.texture_loss in (
                "custom",
                "SpectralPeriodLoss",
                "LogPolarAlignLoss",
                "ScaleConsistencyLoss",
                "Phase",
                "local"
            )
        ):
            vae = self.sd.vae
            scaling_factor = vae.config.get("scaling_factor", 1.0)

            tgt_latents_vae = batch.latents.to(vae.device, dtype=vae.dtype)

            alpha = 0.1

            step = self.current_step
            warmup_steps = 2000

            latents_for_pred = batch.latents.to(
                noise_pred.device, dtype=noise_pred.dtype
            )
            c_lat = latents_for_pred.shape[1]
            c_pred = noise_pred.shape[1]

            if c_pred == c_lat:
                proj = noise_pred
            elif c_pred > c_lat:
                proj = noise_pred[:, :c_lat, :, :]
            else:
                pad_ch = c_lat - c_pred
                pad = torch.zeros(
                    noise_pred.shape[0],
                    pad_ch,
                    noise_pred.shape[2],
                    noise_pred.shape[3],
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                proj = torch.cat([noise_pred, pad], dim=1)

            pred_latents_vae = (latents_for_pred + alpha * proj).to(
                vae.device, dtype=vae.dtype 
            )

            pred_imgs = vae.decode(pred_latents_vae / scaling_factor).sample
            tgt_imgs = vae.decode(tgt_latents_vae / scaling_factor).sample

            # <<< ВАЖНО: перевести в float32 для FFT >>>
            pred_imgs = pred_imgs.float()
            tgt_imgs = tgt_imgs.float()

            if batch.mask_tensor is not None and len(pred_imgs.shape) == 4:
                print('mask_full')
                mask_img = batch.mask_tensor.to(
                    vae.device, dtype=pred_imgs.dtype
                )
                mask_img = torch.nn.functional.interpolate(
                    mask_img,
                    size=pred_imgs.shape[-2:],
                    mode="bicubic",
                    align_corners=False,
                )
            else:
                if len(pred_imgs.shape) == 4:
                    print('mask_one')
                    mask_img = torch.ones(
                        pred_imgs.shape[0],
                        1,
                        pred_imgs.shape[2],
                        pred_imgs.shape[3],
                        device=vae.device,
                        dtype=pred_imgs.dtype,
                    )
                else:
                    mask_img = None


            
            if mask_img is not None:
                mask_img = mask_img.float()
                beta = min(1.0, float(step) / float(warmup_steps))
                final_beta = self.train_config.min_beta + beta * (self.train_config.max_beta - self.train_config.min_beta)
                if self.train_config.texture_loss == "custom":
                    print("custom with coef")
                    tex_sp = self.spectral_period_loss(pred_imgs, tgt_imgs, mask_img)
                    tex_lp = self.logpolar_loss(pred_imgs, tgt_imgs, mask_img)
                    tex_acf = self.acf_period_loss(pred_imgs, tgt_imgs, mask_img)
                    tex_ph = self.phase_loss(pred_imgs, tgt_imgs, mask_img)
                    additional_loss = additional_loss + (self.train_config.loss_Spect_coef * tex_sp + self.train_config.loss_Log_coef * tex_lp + self.train_config.loss_AFC_coef * tex_acf + self.train_config.loss_Phase_coef * tex_ph)
                    if step % 50 == 0:  # лог
                        diag = self.phase_loss.diagnostics(pred_imgs, tgt_imgs, mask_img)
                        print(
                            f"[step {step}] PCL diag: "
                            f"center_prob={diag['center_prob']:.3f}  PCE={diag['PCE']:.2f}"
                        )
                elif self.train_config.texture_loss == "Phase":
                    print("Phase")
                    tex_ph = self.phase_loss(pred_imgs, tgt_imgs, mask_img)
                    additional_loss = additional_loss + tex_ph
                    if step % 50 == 0:  # лог
                        diag = self.phase_loss.diagnostics(pred_imgs, tgt_imgs, mask_img)
                        print(
                            f"[step {step}] PCL diag: "
                            f"center_prob={diag['center_prob']:.3f}  PCE={diag['PCE']:.2f}")
                elif self.train_config.texture_loss == "SpectralPeriodLoss":
                    print("SpectralPeriodLoss")
                    tex_sp = self.spectral_period_loss(pred_imgs, tgt_imgs, mask_img)
                    additional_loss = additional_loss +  tex_sp

                elif self.train_config.texture_loss == "LogPolarAlignLoss":
                    print("LogPolarAlignLoss")
                    tex_lp = self.logpolar_loss(pred_imgs, tgt_imgs, mask_img)
                    additional_loss = additional_loss + tex_lp

                elif self.train_config.texture_loss == "ACFPeriodLoss":
                    print("ACFPerdiodLoss")
                    tex_acf = self.acf_period_loss(pred_imgs, tgt_imgs, mask_img)
                    additional_loss = additional_loss + tex_acf

                elif self.train_config.texture_loss == "local":
                    print("local")
                    tex_local = self.local_texture_loss(pred_imgs, tgt_imgs, mask_img)
                    additional_loss = additional_loss + tex_local
                print(f"BETA: {final_beta}")
                print(f"LOSS {loss} and {additional_loss}")
                return loss + final_beta * additional_loss
        return loss