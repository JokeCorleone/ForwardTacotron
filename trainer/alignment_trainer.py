import time
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CTCLoss
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from models.aligner import Aligner
from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1
from utils import hparams as hp
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel
from utils.dsp import reconstruct_waveform, rescale_mel, np_now
from utils.paths import Paths
from utils.text import sequence_to_text


class AlignmentTrainer:

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.ctc_loss = CTCLoss()

    def train(self, model: Aligner, optimizer: Optimizer) -> None:
        for i, session_params in enumerate(hp.aligner_schedule, 1):
            lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=1, model_type='aligner')
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session)

    def train_session(self, model: ForwardTacotron,
                      optimizer: Optimizer, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])

        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, (x, m, ids, mel_lens, seq_lens) in enumerate(session.train_set, 1):

                start = time.time()
                model.train()
                x, m, mel_lens, seq_lens = x.to(device), m.to(device), \
                                                mel_lens.to(device), seq_lens.to(device)

                m = m.transpose(1, 2)
                model.train()
                pred = model(m)
                pred = pred.transpose(0, 1).log_softmax(2)
                loss = self.ctc_loss(pred, x, mel_lens, seq_lens)
                loss_avg.add(loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                first_pred = pred.transpose(0, 1)[0].max(1)[1].detach().cpu().numpy().tolist()
                first_pred_d = sequence_to_text(first_pred)
                first_target = x[0].detach().cpu().numpy().tolist()
                first_target_d = sequence_to_text(first_target)

                if model.get_step() % 1000 == 0:
                    print()
                    print(f'pred dec: {first_pred_d}')
                    print(f'target dec: {first_target_d}')

                if step % hp.forward_checkpoint_every == 0:
                    ckpt_name = f'forward_step{k}K'
                    save_checkpoint('forward', self.paths, model, optimizer,
                                    name=ckpt_name, is_silent=True)

                if step % hp.forward_plot_every == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            save_checkpoint('aligner', self.paths, model, optimizer, is_silent=True)
            loss_avg.reset()
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
                m1_hat, m2_hat, dur_hat = model(x, m, dur)
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

        m1_hat, m2_hat, dur_hat = model(x, m, dur)
        m1_hat = np_now(m1_hat)[0, :600, :]
        m2_hat = np_now(m2_hat)[0, :600, :]
        m = np_now(m)[0, :600, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        m_fig = plot_mel(m)

        self.writer.add_figure('Ground_Truth_Aligned/target', m_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.step)

        m1_hat, m2_hat, m = rescale_mel(m1_hat), rescale_mel(m2_hat), rescale_mel(m)
        m2_hat_wav = reconstruct_waveform(m2_hat)
        target_wav = reconstruct_waveform(m)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=hp.sample_rate)

        m1_hat, m2_hat, dur_hat = model.generate(x[0].tolist())
        m1_hat, m2_hat = rescale_mel(m1_hat), rescale_mel(m2_hat)
        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)

        self.writer.add_figure('Generated/target', m_fig, model.step)
        self.writer.add_figure('Generated/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Generated/postnet', m2_hat_fig, model.step)

        m2_hat_wav = reconstruct_waveform(m2_hat)

        self.writer.add_audio(
            tag='Generated/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
        self.writer.add_audio(
            tag='Generated/postnet_wav', snd_tensor=m2_hat_wav,
            global_step=model.step, sample_rate=hp.sample_rate)
