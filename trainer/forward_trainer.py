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
from utils.dsp import reconstruct_waveform, rescale_mel, np_now, label_2_float
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
            for i, (x, m, ids, lens, dur, y) in enumerate(session.train_set, 1):
                start = time.time()
                model.train()
                model.gen.train()
                model.disc.train()
                x, m, dur, lens, y = x.to(device), m.to(device), dur.to(device), lens.to(device), y.to(device)

                y_hat, dur_hat, x_out = model.gen(x, m, dur, y)

                # train generator
                model.zero_grad()
                gen_opti.zero_grad()
                y = label_2_float(y, 16)
                y = y.float().unsqueeze(1)
                feats_fake, score_fake = model.disc(y_hat)
                feats_real, score_real = model.disc(y)

                loss_g = 0.0

                loss_g += 10.*torch.mean(torch.mean(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))


                for feat_f, feat_r in zip(feats_fake, feats_real):
                    loss_g += 10. * torch.mean(torch.abs(feat_f - feat_r))

                dur_loss = F.l1_loss(dur_hat, dur)
                loss_g += dur_loss
                loss_g.backward()

                torch.nn.utils.clip_grad_norm_(model.gen.parameters(), 1.0)
                gen_opti.step()

                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                # train discriminator
                y_hat = y_hat.detach()
                loss_d_sum = 0.0
                disc_opti.zero_grad()
                _, score_fake = model.disc(y_hat)
                _, score_real = model.disc(y)
                loss_d = 0.0

                loss_d += 10.*torch.mean(torch.mean(torch.pow(score_real - 1.0, 2), dim=[1, 2]))
                loss_d += 10.*torch.mean(torch.mean(torch.pow(score_fake, 2), dim=[1, 2]))

                loss_d.backward()
                disc_opti.step()
                loss_d_sum += loss_d

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) '\
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
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            save_checkpoint('forward', self.paths, model, gen_opti, disc_opti, is_silent=True)

            m_loss_avg.reset()
            duration_avg.reset()
            print(' ')

    @ignore_exception
    def generate_plots(self, model: ForwardTacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        x, m, ids, lens, dur, y = session.val_sample
        x, m, dur, y = x.to(device), m.to(device), dur.to(device), y.to(device)
        y_hat, dur_hat = model.gen.generate(x[0].tolist())
        y = label_2_float(y, 16)
        y = np_now(y[0])
        self.writer.add_audio(
            tag='Samples/target_wav', snd_tensor=y,
            global_step=model.get_step(), sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Samples/generated_wav', snd_tensor=y_hat,
            global_step=model.get_step(), sample_rate=hp.sample_rate)

