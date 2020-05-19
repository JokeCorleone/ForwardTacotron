import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel
from utils.distribution import MaskedBCE
from utils.dsp import reconstruct_waveform, rescale_mel, np_now
from utils.paths import Paths


class ForwardTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()
        self.disc_loss = MaskedBCE()

    def train(self, model, gen_opti, disc_opti) -> None:
        for i, session_params in enumerate(hp.forward_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='forward')
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, gen_opti, disc_opti, session)

    def train_session(self, model,
                      gen_opti, disc_opti, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in gen_opti.param_groups:
            g['lr'] = session.lr

        for g in disc_opti.param_groups:
            g['lr'] = session.lr

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (x, m, ids, lens, dur) in enumerate(session.train_set, 1):
                start = time.time()
                model.train()
                model.gen.train()
                model.disc.train()
                x, m, dur, lens = x.to(device), m.to(device), dur.to(device), lens.to(device)

                m1_hat, m2_hat, dur_hat, x_out = model.gen(x, m, dur)

                # train generator
                model.zero_grad()
                gen_opti.zero_grad()
                feats_fake, score_fake = model.disc(m2_hat)
                feats_real, score_real = model.disc(m)

                loss_g = 0.0

                loss_g += torch.mean(torch.pow(score_fake - 1.0, 2), dim=[1, 2])
                for feat_f, feat_r in zip(feats_fake, feats_real):
                    loss_g += 10. * torch.mean(torch.abs(feat_f - feat_r))

                dur_loss = F.l1_loss(dur_hat, dur)
                m_loss = F.l1_loss(m2_hat, m)

                #loss_g += 0.*m_loss
                loss_g += dur_loss

                loss_g.backward()

                torch.nn.utils.clip_grad_norm_(model.gen.parameters(), 1.0)
                gen_opti.step()

                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                # train discriminator
                m2_hat = m2_hat.detach()
                loss_d_sum = 0.0
                disc_opti.zero_grad()
                _, score_fake = model.disc(m2_hat)
                _, score_real = model.disc(m)
                loss_d = 0.0

                loss_d += torch.mean(torch.sum(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                loss_d += torch.mean(torch.sum(torch.pow(score_fake, 2), dim=[1, 2]))

                loss_d.backward()
                disc_opti.step()
                loss_d_sum += loss_d

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss {m_loss.item():#.4} ' \
                      f'| Gen Loss {loss_g.item():#.4}' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | {speed:#.2} steps/s | Step: {k}k | '

                if step % hp.forward_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('forward', self.paths, model, gen_opti, disc_opti,
                                    name=ckpt_name, is_silent=True)

                if step % hp.forward_plot_every == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Gan/gen', loss_g, model.get_step())
                self.writer.add_scalar('Gan/disc', loss_d, model.get_step())
                self.writer.add_scalar('Mel_Loss/train', m_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            m_val_loss, dur_val_loss = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', m_val_loss, model.get_step())
            self.writer.add_scalar('Duration_Loss/val', dur_val_loss, model.get_step())
            save_checkpoint('forward', self.paths, model, gen_opti, disc_opti, is_silent=True)

            m_loss_avg.reset()
            duration_avg.reset()
            print(' ')

    def evaluate(self, model: ForwardTacotron, val_set: Dataset) -> Tuple[float, float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        device = next(model.parameters()).device
        for i, (x, m, ids, lens, dur) in enumerate(val_set, 1):
            x, m, dur, lens = x.to(device), m.to(device), dur.to(device), lens.to(device)
            with torch.no_grad():
                m1_hat, m2_hat, dur_hat, x_out = model.gen(x, m, dur)
                m1_loss = self.l1_loss(m1_hat, m, lens)
                m2_loss = self.l1_loss(m2_hat, m, lens)
                dur_loss = F.l1_loss(dur_hat, dur)
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
        return m_val_loss / len(val_set), dur_val_loss / len(val_set)

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        x, m, ids, lens, dur = session.val_sample
        x, m, dur = x.to(device), m.to(device), dur.to(device)

        m1_hat, m2_hat, dur_hat, x_out = model.gen(x, m, dur)
        m1_hat = np_now(m1_hat)[0, :600, :]
        m2_hat = np_now(m2_hat)[0, :600, :]
        m = np_now(m)[0, :600, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_fig = plot_mel(m)

        self.writer.add_figure('Ground_Truth_Aligned/target', m_fig, model.get_step())
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.get_step())
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.get_step())

        m1_hat, m2_hat, m = rescale_mel(m1_hat), rescale_mel(m2_hat), rescale_mel(m)
        m2_hat_wav = reconstruct_waveform(m2_hat)
        target_wav = reconstruct_waveform(m)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.get_step(), sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.get_step(), sample_rate=hp.sample_rate)

        m1_hat, m2_hat, dur_hat = model.gen.generate(x[0].tolist())
        m1_hat, m2_hat = rescale_mel(m1_hat), rescale_mel(m2_hat)
        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)

        self.writer.add_figure('Generated/target', m_fig, model.get_step())
        self.writer.add_figure('Generated/linear', m1_hat_fig, model.get_step())
        self.writer.add_figure('Generated/postnet', m2_hat_fig, model.get_step())

        m2_hat_wav = reconstruct_waveform(m2_hat)

        self.writer.add_audio(
            tag='Generated/target_wav', snd_tensor=target_wav,
            global_step=model.get_step(), sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Generated/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.get_step(), sample_rate=hp.sample_rate)
