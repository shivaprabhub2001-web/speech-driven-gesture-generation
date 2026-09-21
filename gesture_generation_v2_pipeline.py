# Speech-driven gesture generation - model v2 pipeline
# Run each section as a separate Google Colab cell, in order (GPU runtime).
# Lines starting with '!' are Colab shell commands.


# ================= CELL A: time-aligned word features (run once, ~15-25 min) =================
!pip -q install transformers huggingface_hub
import os, glob, numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import BertTokenizerFast, BertModel

root = '/content/drive/MyDrive/ BEAT2_data'
names = sorted(os.path.basename(f).replace('_processed.npz', '') for f in glob.glob(root + '/processed_cache/*_processed.npz'))
prefixes = sorted({'_'.join(n.split('_')[:2]) for n in names})          # e.g. 2_scott
print(len(names), 'sequences | speakers:', prefixes)

# Download only the TextGrids (word timings) for our speakers, plus the official split
local = snapshot_download('H-Liu1997/BEAT2', repo_type='dataset',
                          allow_patterns=[f'beat_english_v2.0.0/textgrid/{p}_*' for p in prefixes]
                                         + ['beat_english_v2.0.0/train_test_split.csv'])
tg_dir = f'{local}/beat_english_v2.0.0/textgrid'

def read_words(path):
    """Return [(start_sec, end_sec, word)] from the word tier of a Praat TextGrid."""
    tiers, cur, iv = {}, None, {}
    for line in open(path, encoding='utf-8', errors='ignore'):
        s = line.strip()
        if s.startswith('name ='):
            cur = s.split('=', 1)[1].strip().strip('"'); tiers[cur] = []; iv = {}
        elif cur is None:
            continue
        elif s.startswith('xmin ='):
            iv['xmin'] = float(s.split('=', 1)[1])
        elif s.startswith('xmax ='):
            iv['xmax'] = float(s.split('=', 1)[1])
        elif s.startswith('text ='):
            t = s.split('=', 1)[1].strip().strip('"')
            if t.strip() and 'xmin' in iv and 'xmax' in iv:
                tiers[cur].append((iv['xmin'], iv['xmax'], t))
            iv = {}
    if not tiers:
        return []
    key = next((k for k in tiers if 'word' in k.lower()), next(iter(tiers)))
    return tiers[key]

tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').cuda().eval()

@torch.no_grad()
def word_embeddings(words, chunk=150):
    """One 768-d BERT vector per word (average of its sub-word tokens)."""
    out = np.zeros((len(words), 768), np.float32)
    for s in range(0, len(words), chunk):
        w = words[s:s + chunk]
        enc = tok(w, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=512)
        h = bert(**{k: v.cuda() for k, v in enc.items()}).last_hidden_state[0].cpu().numpy()
        ids = enc.word_ids()
        for j in range(len(w)):
            idx = [k for k, wid in enumerate(ids) if wid == j]
            if idx:
                out[s + j] = h[idx].mean(0)
    return out

out_dir = f'{root}/word_features'; os.makedirs(out_dir, exist_ok=True)
missing = []
for i, n in enumerate(names, 1):
    out = f'{out_dir}/{n}.npz'
    if os.path.exists(out):
        continue                                   # resumable if Colab disconnects
    tg = f'{tg_dir}/{n}.TextGrid'
    words = read_words(tg) if os.path.exists(tg) else []
    if not words:
        missing.append(n); continue
    emb = word_embeddings([w for _, _, w in words])
    np.savez(out, emb=emb.astype(np.float16),
             starts=np.array([s for s, _, _ in words], np.float32),
             ends=np.array([e for _, e, _ in words], np.float32))
    if i % 25 == 0:
        print(f'  {i}/{len(names)} done')

print('Word features saved:', len(glob.glob(out_dir + '/*.npz')), '| missing TextGrids:', missing[:10], len(missing))
import shutil; shutil.copy(f'{local}/beat_english_v2.0.0/train_test_split.csv', f'{root}/train_test_split.csv')


# ================= COPY DATA TO COLAB'S LOCAL DISK (run once per session, ~2-5 min) =================
import os, glob, shutil, time, numpy as np

# Make sure Drive is connected (remount if the connection dropped)
from google.colab import drive
if not os.path.ismount('/content/drive'):
    if os.path.exists('/content/drive'):
        shutil.rmtree('/content/drive')
    drive.mount('/content/drive')
else:
    drive.mount('/content/drive', force_remount=True)

root = '/content/drive/MyDrive/ BEAT2_data'
DATA = '/content/beat_local'
for sub in ['processed_cache', 'word_features']:
    os.makedirs(f'{DATA}/{sub}', exist_ok=True)
shutil.copy(f'{root}/train_test_split.csv', f'{DATA}/train_test_split.csv')

def copy_with_retry(src, dst, tries=5):
    for k in range(tries):
        try:
            shutil.copy(src, dst)
            np.load(dst)                     # check the copy is a valid file
            return True
        except Exception as e:
            if k == tries - 1:
                print('  FAILED:', os.path.basename(src), '-', e)
                return False
            time.sleep(3)

t0 = time.time()
for sub in ['processed_cache', 'word_features']:
    srcs = sorted(glob.glob(f'{root}/{sub}/*.npz'))
    done = 0
    for i, s in enumerate(srcs, 1):
        d = f'{DATA}/{sub}/{os.path.basename(s)}'
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            done += 1; continue
        done += copy_with_retry(s, d)
        if i % 100 == 0:
            print(f'  {sub}: {i}/{len(srcs)}')
    print(f'{sub}: {done}/{len(srcs)} files copied OK')
print(f'Finished in {time.time()-t0:.0f}s')


# ================= CELL B: improved model v2 + training (~20-40 min on T4) =================
import os, glob, math, time, shutil, numpy as np, pandas as pd, torch, torch.nn as nn

root = '/content/drive/MyDrive/ BEAT2_data'      # checkpoints are saved here
DATA = '/content/beat_local'                     # data is read from Colab's local disk (fast, no Drive dropouts)
DEVICE = torch.device('cuda')
CKPT_DIR = f'{root}/checkpoints_v2'; os.makedirs(CKPT_DIR, exist_ok=True)

FPS, AUDIO_FPS, WIN = 30, 16000 / 512, 200      # motion 30 fps; cached audio features 31.25 fps
D, HEADS, N_ENC, N_DEC = 256, 4, 4, 4
STEPS = 200                                     # diffusion steps (same as v1)
EPOCHS, BATCH, LR, WINDOWS_PER_FILE, LAMBDA_VEL = 200, 32, 1e-4, 4, 1.0
torch.manual_seed(42); np.random.seed(42)

# ---------- 1. Load data into RAM, with the two alignment fixes ----------
split = pd.read_csv(f'{DATA}/train_test_split.csv')
split_type = dict(zip(split['id'], split['type']))
names = sorted(os.path.basename(f).replace('_processed.npz', '') for f in glob.glob(DATA + '/processed_cache/*_processed.npz'))

def load_item(name):
    d = np.load(f'{DATA}/processed_cache/{name}_processed.npz')
    motion = d['motion'].astype(np.float32); T = len(motion)
    a = d['audio_features'].astype(np.float32)
    # FIX 1: resample audio features from 31.25 fps to the 30 fps motion timeline
    src_t, tgt_t = np.arange(len(a)) / AUDIO_FPS, np.arange(T) / FPS
    audio = np.stack([np.interp(tgt_t, src_t, a[:, k]) for k in range(a.shape[1])], 1).astype(np.float32)
    # FIX 2: place each word's BERT vector on the frames where that word is spoken
    w = np.load(f'{DATA}/word_features/{name}.npz')
    word_idx = np.full(T, -1, np.int32)
    for i, (s, e) in enumerate(zip(w['starts'], w['ends'])):
        fs = int(round(s * FPS)); fe = max(int(round(e * FPS)), fs + 1)
        word_idx[fs:fe] = i
    return dict(name=name, motion=motion, audio=audio, word_emb=w['emb'].astype(np.float32), word_idx=word_idx)

t0 = time.time()
items = [load_item(n) for n in names if os.path.exists(f'{DATA}/word_features/{n}.npz')]
items = [it for it in items if len(it['motion']) >= WIN]
# FIX 3: proper held-out split -> test files are NOT used for training
train = [it for it in items if split_type.get(it['name'], 'train') in ('train', 'additional')]
val   = [it for it in items if split_type.get(it['name']) == 'val']
test  = [it for it in items if split_type.get(it['name']) == 'test']
if len(val) == 0:                                   # no official val files for these speakers -> hold out 5% of train
    rng = np.random.RandomState(0); idx = rng.permutation(len(train)); k = max(1, len(train) // 20)
    val = [train[i] for i in idx[:k]]; train = [train[i] for i in idx[k:]]
print(f'Loaded {len(items)} sequences in {time.time()-t0:.0f}s | train {len(train)} | val {len(val)} | test {len(test)}')

# Normalisation statistics from the TRAINING set only
cat = lambda k: np.concatenate([it[k] for it in train])
stats = {k: v.astype(np.float32) for k, v in dict(
    m_mean=cat('motion').mean(0), m_std=np.maximum(cat('motion').std(0), 1e-3),
    a_mean=cat('audio').mean(0),  a_std=np.maximum(cat('audio').std(0), 1e-3)).items()}

def make_window(it, start):
    sl = slice(start, start + WIN)
    motion = (it['motion'][sl] - stats['m_mean']) / stats['m_std']
    audio = (it['audio'][sl] - stats['a_mean']) / stats['a_std']
    idx = it['word_idx'][sl]
    words = np.zeros((WIN, 768), np.float32); has = idx >= 0
    words[has] = it['word_emb'][idx[has]]
    return (torch.from_numpy(motion), torch.from_numpy(audio), torch.from_numpy(words))

def batches(data, n_per_file, random_start):
    order = np.random.permutation(len(data) * n_per_file) if random_start else np.arange(len(data))
    for b in range(0, len(order), BATCH):
        ws = []
        for j in order[b:b + BATCH]:
            it = data[j % len(data)]
            # FIX 4: random windows from the whole recording (v1 only ever used the first 200 frames)
            s = np.random.randint(0, len(it['motion']) - WIN + 1) if random_start else 0
            ws.append(make_window(it, s))
        yield [torch.stack(x).to(DEVICE) for x in zip(*ws)]

# ---------- 2. Model v2 ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4000):
        super().__init__()
        pos = torch.arange(max_len)[:, None]; div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.shape[1]][None]

def timestep_embedding(t, d):
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)

class SpeechEncoderV2(nn.Module):
    """Frame-aligned audio + word features, WITH positional encoding (v1 had none)."""
    def __init__(self):
        super().__init__()
        self.audio_proj, self.word_proj = nn.Linear(14, D), nn.Linear(768, D)
        self.pe = PositionalEncoding(D)
        layer = nn.TransformerEncoderLayer(D, HEADS, D * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, N_ENC)
    def forward(self, audio, words):
        return self.enc(self.pe(self.audio_proj(audio) + self.word_proj(words)))

class MotionDenoiserV2(nn.Module):
    """FIX 5: temporal self-attention between motion frames + cross-attention to speech, stacked 4x."""
    def __init__(self, motion_dim=165):
        super().__init__()
        self.in_proj, self.out_proj = nn.Linear(motion_dim, D), nn.Linear(D, motion_dim)
        self.t_mlp = nn.Sequential(nn.Linear(D, D), nn.SiLU(), nn.Linear(D, D))
        self.pe = PositionalEncoding(D)
        layer = nn.TransformerDecoderLayer(D, HEADS, D * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, N_DEC)
    def forward(self, x_t, t, cond):
        h = self.pe(self.in_proj(x_t) + cond) + self.t_mlp(timestep_embedding(t, D))[:, None]
        return self.out_proj(self.dec(tgt=h, memory=cond))      # predicts the clean motion x0

# FIX 6: cosine noise schedule. v1's linear schedule over 200 steps ended at alpha_bar = 0.13,
# so sampling started from pure noise that the model had never been trained on.
def cosine_betas(T, s=0.008):
    x = np.linspace(0, T, T + 1)
    ac = np.cos(((x / T) + s) / (1 + s) * np.pi / 2) ** 2
    return torch.tensor(np.clip(1 - ac[1:] / ac[:-1], 0, 0.999), dtype=torch.float32)

betas = cosine_betas(STEPS).to(DEVICE); alphas = 1 - betas; alphas_cumprod = torch.cumprod(alphas, 0)
print('alpha_bar at final step: %.6f (should be close to 0)' % alphas_cumprod[-1].item())

encoder, denoiser = SpeechEncoderV2().to(DEVICE), MotionDenoiserV2().to(DEVICE)
params = list(encoder.parameters()) + list(denoiser.parameters())
print('Parameters: %.1fM' % (sum(p.numel() for p in params) / 1e6))
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
scaler = torch.amp.GradScaler('cuda')

def diffusion_loss(motion, audio, words, gen=None):
    t = torch.randint(0, STEPS, (motion.shape[0],), device=DEVICE, generator=gen)
    noise = torch.randn(motion.shape, device=DEVICE, generator=gen)
    ab = alphas_cumprod[t].view(-1, 1, 1)
    x_t = ab.sqrt() * motion + (1 - ab).sqrt() * noise
    with torch.amp.autocast('cuda'):
        pred = denoiser(x_t, t, encoder(audio, words))
    pred = pred.float()
    # FIX 7: velocity loss penalises frame-to-frame jitter directly
    return nn.functional.mse_loss(pred, motion) + LAMBDA_VEL * nn.functional.mse_loss(pred.diff(dim=1), motion.diff(dim=1))

# ---------- 3. Train (resumes automatically if Colab disconnects) ----------
start_epoch, best_val, history = 1, float('inf'), []
latest = f'{CKPT_DIR}/latest.pt'
if os.path.exists(latest):
    ck = torch.load(latest, map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ck['encoder']); denoiser.load_state_dict(ck['denoiser'])
    opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
    start_epoch, best_val, history = ck['epoch'] + 1, ck['best_val'], ck['history']
    print('Resuming from epoch', ck['epoch'])

def save(path, epoch):
    # save locally first, then copy to Drive; a Drive dropout will not stop training
    local_path = '/content/' + os.path.basename(path)
    torch.save(dict(encoder=encoder.state_dict(), denoiser=denoiser.state_dict(), opt=opt.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, best_val=best_val, history=history, stats=stats,
                    config=dict(D=D, HEADS=HEADS, N_ENC=N_ENC, N_DEC=N_DEC, STEPS=STEPS, WIN=WIN, FPS=FPS)), local_path)
    try:
        shutil.copy(local_path, path)
    except Exception as e:
        print('  (warning: could not copy checkpoint to Drive this time:', e, ')')

for epoch in range(start_epoch, EPOCHS + 1):
    encoder.train(); denoiser.train(); losses = []; t0 = time.time()
    for motion, audio, words in batches(train, WINDOWS_PER_FILE, True):
        loss = diffusion_loss(motion, audio, words)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); losses.append(loss.item())

    encoder.eval(); denoiser.eval()
    g = torch.Generator(device=DEVICE).manual_seed(0)           # same noise every epoch -> comparable val loss
    with torch.no_grad():
        val_loss = float(np.mean([diffusion_loss(m, a, w, g).item() for m, a, w in batches(val, 1, False)]))
    history.append((epoch, float(np.mean(losses)), val_loss))
    print(f'Epoch {epoch:3d}/{EPOCHS} | train {np.mean(losses):.4f} | val {val_loss:.4f} | {time.time()-t0:.0f}s')

    if val_loss < best_val:
        best_val = val_loss; save(f'{CKPT_DIR}/best.pt', epoch)
    if epoch % 10 == 0 or epoch == EPOCHS:
        save(latest, epoch)

print('Training complete. Best val loss %.4f' % best_val)


# ================= CELL C: evaluate v1 vs v2 on held-out TEST files (~5-10 min) =================
# Needs the objects from the copy cell + Cell B in this session.
# If Colab restarted: mount/copy cell -> Cell B (it sees 200 epochs are done and skips training) -> this cell.
!pip -q install librosa soundfile
import os, numpy as np, pandas as pd, torch, torch.nn as nn, librosa, matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import find_peaks
from huggingface_hub import hf_hub_download

OUT = f'{root}/results_v2'; os.makedirs(OUT, exist_ok=True)
GEN_DIR = f'{root}/generated_samples_v2'; os.makedirs(GEN_DIR, exist_ok=True)

# ---------- Load the best v2 checkpoint ----------
ck = torch.load(f'{CKPT_DIR}/best.pt', map_location=DEVICE, weights_only=False)
enc2, den2 = SpeechEncoderV2().to(DEVICE), MotionDenoiserV2().to(DEVICE)
enc2.load_state_dict(ck['encoder']); den2.load_state_dict(ck['denoiser']); enc2.eval(); den2.eval()
st = ck['stats']
print('v2 best checkpoint: epoch', ck['epoch'], '| val loss %.4f' % ck['best_val'])

@torch.no_grad()
def sample_v2(it, seed):
    _, audio_n, words = make_window(it, 0)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    cond = enc2(audio_n[None].to(DEVICE), words[None].to(DEVICE))
    x = torch.randn((1, WIN, 165), device=DEVICE, generator=g)
    for i in reversed(range(STEPS)):
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        x0 = den2(x, t, cond)
        if i > 0:                                              # DDPM posterior q(x_{t-1} | x_t, x0)
            ab_t, ab_prev = alphas_cumprod[i], alphas_cumprod[i - 1]
            mean = (ab_prev.sqrt() * betas[i] / (1 - ab_t)) * x0 + (alphas[i].sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
            x = mean + (betas[i] * (1 - ab_prev) / (1 - ab_t)).sqrt() * torch.randn(x.shape, device=DEVICE, generator=g)
        else:
            x = x0
    return x[0].cpu().numpy() * st['m_std'] + st['m_mean']

# ---------- Rebuild v1 exactly as trained ----------
class SpeechEncoderV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_proj, self.bert_proj = nn.Linear(14, 256), nn.Linear(768, 256)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, batch_first=True), 4)
    def forward(self, audio, bert):
        a, b = self.audio_proj(audio), self.bert_proj(bert); n = min(a.shape[1], b.shape[1])
        return self.transformer(a[:, :n] + b[:, :n])

class DiffusionDecoderV1(nn.Module):
    def __init__(self, motion_dim=165):
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, 256))
        self.motion_proj = nn.Linear(motion_dim, 256)
        self.cross_attn = nn.MultiheadAttention(256, 4, batch_first=True)
        self.norm1 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 512), nn.SiLU(), nn.Linear(512, 256))
        self.norm2 = nn.LayerNorm(256)
        self.out_proj = nn.Linear(256, motion_dim)
    def forward(self, x, t, cond):
        m = self.motion_proj(x) + self.time_embed(t.float().unsqueeze(-1)).unsqueeze(1)
        attn, _ = self.cross_attn(query=m, key=cond, value=cond)
        m = self.norm1(m + attn); m = self.norm2(m + self.ffn(m))
        return self.out_proj(m)

ck1 = torch.load(f'{root}/checkpoints/model_epoch30.pt', map_location=DEVICE, weights_only=False)
enc1, dec1 = SpeechEncoderV1().to(DEVICE), DiffusionDecoderV1().to(DEVICE)
enc1.load_state_dict(ck1['encoder']); dec1.load_state_dict(ck1['decoder']); enc1.eval(); dec1.eval()
b1 = torch.linspace(1e-4, 0.02, 200, device=DEVICE); a1 = 1 - b1; ac1 = torch.cumprod(a1, 0)
fix_len = lambda arr, n: arr[:n] if len(arr) >= n else np.pad(arr, [(0, n - len(arr))] + [(0, 0)] * (arr.ndim - 1))

@torch.no_grad()
def sample_v1(name, seed):
    d = np.load(f'{DATA}/processed_cache/{name}_processed.npz')
    a = torch.tensor(fix_len(d['audio_features'], 200), dtype=torch.float32, device=DEVICE)[None]
    b = torch.tensor(fix_len(d['bert_features'], 200), dtype=torch.float32, device=DEVICE)[None]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    cond = enc1(a, b); x = torch.randn((1, 200, 165), device=DEVICE, generator=g)
    for i in reversed(range(200)):
        eps = dec1(x, torch.full((1,), i, device=DEVICE, dtype=torch.long), cond)
        noise = torch.randn(x.shape, device=DEVICE, generator=g) if i > 0 else torch.zeros_like(x)
        x = (1 / a1[i].sqrt()) * (x - ((1 - a1[i]) / (1 - ac1[i]).sqrt()) * eps) + b1[i].sqrt() * noise
    return x[0].cpu().numpy()

# ---------- Metrics (identical for real, v1 and v2) ----------
UB = slice(3, 66)                                          # upper-body joint rotations
jitter = lambda m: np.abs(np.diff(m[:, UB], axis=0)).mean()

def stat_feat(m):                                          # same statistical FGD features as the original study
    v = np.diff(m, axis=0)
    return np.concatenate([m.mean(0), m.std(0), v.mean(0), v.std(0)])

def fgd(R, G):
    mu_r, mu_g = R.mean(0), G.mean(0); s_r, s_g = np.cov(R, rowvar=False), np.cov(G, rowvar=False)
    covmean, _ = linalg.sqrtm(s_r @ s_g, disp=False)
    return float((mu_r - mu_g) @ (mu_r - mu_g) + np.trace(s_r + s_g - 2 * covmean.real))

def beat_align(motion_t, audio_t, sigma=0.1):
    if len(motion_t) == 0 or len(audio_t) == 0:
        return 0.0
    return float(np.mean([np.exp(-np.min(np.abs(motion_t - a)) ** 2 / (2 * sigma ** 2)) for a in audio_t]))

def ba_original(motion, name):
    """Original method: every velocity peak vs peaks in cached audio-feature energy."""
    v = np.linalg.norm(np.diff(motion, axis=0), axis=1)
    m_t = np.array([i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] > v[i + 1]]) / FPS
    e = np.linalg.norm(fix_len(np.load(f'{DATA}/processed_cache/{name}_processed.npz')['audio_features'], 200), axis=1)
    a_t = np.array([i for i in range(1, len(e) - 1) if e[i] > e[i - 1] and e[i] > e[i + 1]]) / FPS
    return beat_align(m_t, a_t)

def ba_improved(motion, onsets):
    """Improved method: audio onsets from the waveform; motion beats = pauses in upper-body speed,
    at least 5 frames apart, so frame-to-frame jitter cannot create hundreds of fake beats."""
    v = np.linalg.norm(np.diff(motion[:, UB], axis=0), axis=1)
    peaks, _ = find_peaks(-v, distance=5)
    return beat_align(peaks / FPS, onsets)

# ---------- Run on every test file ----------
rows, feats = [], {'real': [], 'v1': [], 'v2': []}
for k, it in enumerate(test):
    name = it['name']
    wav = hf_hub_download('H-Liu1997/BEAT2', f'beat_english_v2.0.0/wave16k/{name}.wav', repo_type='dataset')
    y, sr = librosa.load(wav, sr=16000, duration=WIN / FPS)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')

    motions = {'real': it['motion'][:WIN], 'v1': sample_v1(name, seed=42 + k), 'v2': sample_v2(it, seed=42 + k)}
    np.save(f'{GEN_DIR}/generated_{name}.npy', motions['v2'])
    for model, m in motions.items():
        feats[model].append(stat_feat(m))
        rows.append(dict(clip=name, model=model, jitter_ratio=jitter(m) / jitter(motions['real']),
                         pose_std_ratio=m[:, UB].std() / motions['real'][:, UB].std(),
                         beatalign_original=ba_original(m, name), beatalign_improved=ba_improved(m, onsets)))
    if (k + 1) % 10 == 0:
        print(f'  {k+1}/{len(test)} test clips done')

df = pd.DataFrame(rows); df.to_csv(f'{OUT}/per_clip_results.csv', index=False)
summary = df.groupby('model')[['jitter_ratio', 'pose_std_ratio', 'beatalign_original', 'beatalign_improved']].mean()
R = np.stack(feats['real'])
summary['FGD_vs_real'] = [np.nan if m == 'real' else fgd(R, np.stack(feats[m])) for m in summary.index]
summary = summary.loc[['real', 'v1', 'v2']]
summary.to_csv(f'{OUT}/summary_v1_v2.csv')
print(f'\n===== TEST SET RESULTS (N = {len(test)} clips, first {WIN} frames each) =====')
print(summary.round(4).to_string())

# ---------- Training curve figure for the dissertation ----------
h = np.array(ck['history'])
if len(h):
    plt.figure(figsize=(6, 3.5)); plt.plot(h[:, 0], h[:, 1], label='train'); plt.plot(h[:, 0], h[:, 2], label='validation')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Model v2 training'); plt.legend(); plt.tight_layout()
    plt.savefig(f'{OUT}/v2_training_curve.png', dpi=200); plt.show()
print('Saved results to', OUT)


# ================= CELL D: render the six v2 survey clips (~5 min) =================
!pip -q install smplx trimesh pyrender imageio imageio-ffmpeg
!pip -q install --upgrade pyopengl==3.1.7
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import glob, subprocess, numpy as np, torch, smplx, trimesh, pyrender, imageio
from huggingface_hub import hf_hub_download
from IPython.display import Video, display

root = '/content/drive/MyDrive/ BEAT2_data'
FPS = 30
clips = ['2_scott_0_1_1', '4_lawrence_0_1_1', '6_carla_0_65_65', '7_sophie_0_1_1', '2_scott_0_2_2', '4_lawrence_0_2_2']
out_dir = f'{root}/rendered_clips_v2'; os.makedirs(out_dir, exist_ok=True)

model_file = glob.glob('/content/drive/MyDrive/*BEAT2_data/*smplx_models/SMPLX_NEUTRAL*.npz')[0]
body = smplx.SMPLX(model_path=model_file, use_pca=False, num_betas=10, num_expression_coeffs=10, flat_hand_mean=True)

def clean(poses):                     # identical treatment to before: face camera, legs still
    p = poses.copy(); p[:, 0:3] = 0
    for j in [1, 2, 4, 5, 7, 8, 10, 11]:
        p[:, j * 3:(j + 1) * 3] = 0
    return p

def vertices(frame):
    t = torch.tensor(frame, dtype=torch.float32)[None]
    out = body(global_orient=t[:, 0:3], body_pose=t[:, 3:66], jaw_pose=t[:, 66:69], leye_pose=t[:, 69:72],
               reye_pose=t[:, 72:75], left_hand_pose=t[:, 75:120], right_hand_pose=t[:, 120:165])
    return out.vertices[0].detach().numpy()

renderer = pyrender.OffscreenRenderer(640, 720)
top_y = vertices(np.zeros(165))[:, 1].max()
cam_pose = np.eye(4); cam_pose[:3, 3] = [0, top_y - 0.55, 3.0]

def render(frame):
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35, 0.35, 0.35])
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.55, 0.65, 0.85, 1.0], roughnessFactor=0.7)
    scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices(frame), body.faces, process=False), material=mat, smooth=True))
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 6), pose=cam_pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=cam_pose)
    img, depth = renderer.render(scene); img = img.copy(); img[depth == 0] = 255
    return img

speed = lambda m: np.abs(np.diff(m[:, 3:66], axis=0)).mean()
for i, name in enumerate(clips, 1):
    gen = np.load(f'{root}/generated_samples_v2/generated_{name}.npy')          # raw v2 output, NO smoothing
    real = np.load(f'/content/beat_local/processed_cache/{name}_processed.npz')['motion'][:len(gen)] \
        if os.path.exists(f'/content/beat_local/processed_cache/{name}_processed.npz') \
        else np.load(f'{root}/processed_cache/{name}_processed.npz')['motion'][:len(gen)]
    print(f'[{i}/6] {name}: speed ratio vs real {speed(gen)/speed(real):.1f}x')
    silent = f'/content/v2_clip_{i}_silent.mp4'
    with imageio.get_writer(silent, fps=FPS, codec='libx264', quality=8) as w:
        for fr in clean(gen):
            w.append_data(render(fr))
    wav = hf_hub_download('H-Liu1997/BEAT2', f'beat_english_v2.0.0/wave16k/{name}.wav', repo_type='dataset')
    final = f'{out_dir}/clip_{i}_{name}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-ss', '0', '-t', f'{len(gen)/FPS:.3f}', '-i', wav,
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', final], check=True)
    print('   saved:', final)

display(Video(f'{out_dir}/clip_1_{clips[0]}.mp4', embed=True, width=400))


# ================= CELL E: temporal range ratio, Wilcoxon tests (Table 5.3), Figures 5.3-5.4 (~5 min) =================
# Needs the objects from the copy cell + Cell B + Cell C in this session.
# Regenerates v1 and v2 with the same seeds as Cell C (42 + sequence index), so the Table 5.2 values are reproduced,
# and saves every test-set motion used in Sections 5.3-5.5 to results_v2_extra/motions_test.npz.
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon, rankdata

EXTRA = f'{root}/results_v2_extra'; os.makedirs(EXTRA, exist_ok=True)
names_test = [it['name'] for it in test]
N = len(test)

def temporal_range(m):
    """Std over time of each of the 63 body-joint parameters, averaged over the 63."""
    return m[:, UB].std(0).mean()

def ratios(m, real):
    return dict(jitter_ratio=jitter(m) / jitter(real),
                pose_std_ratio=m[:, UB].std() / real[:, UB].std(),
                temporal_range_ratio=temporal_range(m) / temporal_range(real))

# Audio onsets for every test sequence (first WIN frames of the waveform), downloaded once
onsets = {}
for it in test:
    wav = hf_hub_download('H-Liu1997/BEAT2', f'beat_english_v2.0.0/wave16k/{it["name"]}.wav', repo_type='dataset')
    y, sr = librosa.load(wav, sr=16000, duration=WIN / FPS)
    onsets[it['name']] = librosa.onset.onset_detect(y=y, sr=sr, units='time')

def score(m, real, speech_name):
    """All per-sequence measures for motion m, with ratios vs real and beat alignment vs speech_name's audio."""
    return dict(**ratios(m, real), beatalign_original=ba_original(m, speech_name),
                beatalign_improved=ba_improved(m, onsets[speech_name]))

M = {'real': np.stack([it['motion'][:WIN] for it in test]),
     'v1':   np.stack([sample_v1(it['name'], seed=42 + k) for k, it in enumerate(test)]),
     'v2':   np.stack([sample_v2(it, seed=42 + k) for k, it in enumerate(test)])}

def save_motions():
    np.savez_compressed(f'{EXTRA}/motions_test.npz', names=np.array(names_test), **M)

def per_seq(cond, speech_names=None):
    """DataFrame of per-sequence measures for condition cond (scored against own speech unless speech_names given)."""
    speech_names = speech_names or names_test
    return pd.DataFrame([dict(clip=n, **score(M[cond][k], M['real'][k], speech_names[k])) for k, n in enumerate(names_test)])

P = {c: per_seq(c) for c in ['real', 'v1', 'v2']}
save_motions()
print('Means (compare with Table 5.2):')
print(pd.DataFrame({c: P[c].drop(columns='clip').mean() for c in P}).T.round(4).to_string())

def wilcoxon_row(first, second, measure, a, b=None):
    """Paired two-sided Wilcoxon signed-rank test of a - b (or a against 0 if b is None),
    with the matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-)."""
    d = np.asarray(a, float) - (0 if b is None else np.asarray(b, float))
    res = wilcoxon(d)
    nz = d[d != 0]; rk = rankdata(np.abs(nz))
    w_plus, w_minus = rk[nz > 0].sum(), rk[nz < 0].sum()
    return dict(comparison=f'{first} vs {second}', measure=measure, median_difference=float(np.median(d)),
                W=float(res.statistic), p=float(res.pvalue), r=float((w_plus - w_minus) / (w_plus + w_minus)),
                n_first_higher=int((d > 0).sum()), n=len(d))

def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj

dev = lambda x: np.abs(np.log(np.asarray(x)))            # deviation from real = |log(ratio)|
v1, v2, re = P['v1'], P['v2'], P['real']
tests = [
    wilcoxon_row('v1', 'v2', 'Jitter deviation from real', dev(v1.jitter_ratio), dev(v2.jitter_ratio)),
    wilcoxon_row('v1', 'v2', 'Pose spread deviation from real', dev(v1.pose_std_ratio), dev(v2.pose_std_ratio)),
    wilcoxon_row('v2', 'real', 'Pose spread ratio', v2.pose_std_ratio - 1.0),
    wilcoxon_row('v2', 'real', 'Temporal range ratio', v2.temporal_range_ratio - 1.0),
]
for proc in ['original', 'improved']:
    col = f'beatalign_{proc}'
    tests += [wilcoxon_row('v1', 'real', f'BeatAlign, {proc}', v1[col], re[col]),
              wilcoxon_row('v2', 'real', f'BeatAlign, {proc}', v2[col], re[col]),
              wilcoxon_row('v1', 'v2', f'BeatAlign, {proc}', v1[col], v2[col])]
T53 = pd.DataFrame(tests); T53['p_holm'] = holm(T53['p'])
T53.to_csv(f'{EXTRA}/table_5_3_wilcoxon.csv', index=False)
pd.concat([P[c].assign(model=c) for c in P]).to_csv(f'{EXTRA}/per_clip_results_with_temporal_range.csv', index=False)
print('\n===== TABLE 5.3: paired Wilcoxon signed-rank tests (N = %d) =====' % N)
print(T53.round(4).to_string(index=False))

# Figure 5.3: per-sequence distributions
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
panels = [('jitter_ratio', 'Jitter ratio (log scale)', ['v1', 'v2'], True),
          ('pose_std_ratio', 'Pose range ratio', ['v1', 'v2'], False),
          ('beatalign_original', 'BeatAlign, original', ['real', 'v1', 'v2'], False),
          ('beatalign_improved', 'BeatAlign, improved', ['real', 'v1', 'v2'], False)]
jit = np.random.RandomState(0)
for ax, (col, title, conds, logy) in zip(axes, panels):
    for i, c in enumerate(conds):
        vals = P[c][col].values
        ax.scatter(i + jit.uniform(-0.15, 0.15, len(vals)), vals, s=10, alpha=0.6)
        ax.hlines(np.median(vals), i - 0.3, i + 0.3, color='k', lw=2)
    if col.endswith('ratio'):
        ax.axhline(1.0, ls='--', color='grey', lw=1)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds); ax.set_title(title, fontsize=10)
    if logy: ax.set_yscale('log')
plt.tight_layout(); plt.savefig(f'{EXTRA}/figure_5_3_per_sequence.png', dpi=200); plt.show()

# Figure 5.4: right-elbow rotation (SMPL-X joint 19 -> parameters 57:60) for one held-out sequence
clip = '2_scott_0_1_1'
if clip in names_test:
    k = names_test.index(clip)
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True, sharey=True)
    titles = {'real': 'Real motion', 'v1': 'v1 (baseline)', 'v2': 'v2 (improved)'}
    for ax, c in zip(axes, ['real', 'v1', 'v2']):
        ang = np.degrees(np.linalg.norm(M[c][k][:, 57:60], axis=1))
        change = np.abs(np.diff(ang)).mean()
        print(f'{c}: right-elbow mean change {change:.2f} deg/frame')
        ax.plot(np.arange(len(ang)) / FPS, ang, lw=1)
        ax.set_title(f'{titles[c]}   (mean change per frame: {change:.2f}°)', loc='left', fontsize=9)
    axes[1].set_ylabel('Right elbow rotation (degrees)'); axes[-1].set_xlabel('Time (seconds)'); plt.tight_layout()
    plt.savefig(f'{EXTRA}/figure_5_4_right_elbow.png', dpi=200); plt.show()


# ================= CELL F: control experiments, diversity and FGD robustness (Table 5.4) (~25-40 min) =================
# Needs Cell E. Sample s of sequence k uses seed 42 + k + 1000*s, so s = 0 reproduces Table 5.2.
speaker = lambda n: n.split('_')[1]                       # '2_scott_0_1_1' -> 'scott'
by_spk = {}
for k, n in enumerate(names_test):
    by_spk.setdefault(speaker(n), []).append(k)
spk_order = sorted(by_spk)

# Mismatched speech: the next test sequence by the SAME speaker (cyclic, sorted order).
same_partner = {ks[i]: ks[(i + 1) % len(ks)] for ks in by_spk.values() for i in range(len(ks))}
# Different speaker: the sequence in the same position in the NEXT speaker's list (cyclic).
diff_partner = {}
for si, s in enumerate(spk_order):
    other = by_spk[spk_order[(si + 1) % len(spk_order)]]
    for i, k in enumerate(by_spk[s]):
        diff_partner[k] = other[i % len(other)]
mismatched_names = [names_test[same_partner[k]] for k in range(N)]

# Unrelated real recording: a TRAINING recording by the same speaker (first WIN frames), fixed choice
rng = np.random.RandomState(0)
train_by_spk = {}
for it in train:
    train_by_spk.setdefault(speaker(it['name']), []).append(it)
unrelated = [train_by_spk[speaker(n)][rng.randint(len(train_by_spk[speaker(n)]))] for n in names_test]
M['real_unrelated'] = np.stack([u['motion'][:WIN] for u in unrelated])
M['static'] = np.repeat(stats['m_mean'][None, None], N, 0).repeat(WIN, 1)     # training-set mean pose, held still

@torch.no_grad()
def sample_from(enc, den, speech_item, seed, w=None):
    """v2-family DDPM sampler. Speech comes from speech_item; the noise from seed.
    If w is given, classifier-free guidance: x0 = x0_uncond + w * (x0_cond - x0_uncond)."""
    _, audio_n, words = make_window(speech_item, 0)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    cond = enc(audio_n[None].to(DEVICE), words[None].to(DEVICE))
    null = enc(torch.zeros_like(audio_n)[None].to(DEVICE), torch.zeros_like(words)[None].to(DEVICE)) if w is not None else None
    x = torch.randn((1, WIN, 165), device=DEVICE, generator=g)
    for i in reversed(range(STEPS)):
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        x0 = den(x, t, cond)
        if w is not None:
            x0_u = den(x, t, null); x0 = x0_u + w * (x0 - x0_u)
        if i > 0:
            ab_t, ab_prev = alphas_cumprod[i], alphas_cumprod[i - 1]
            mean = (ab_prev.sqrt() * betas[i] / (1 - ab_t)) * x0 + (alphas[i].sqrt() * (1 - ab_prev) / (1 - ab_t)) * x
            x = mean + (betas[i] * (1 - ab_prev) / (1 - ab_t)).sqrt() * torch.randn(x.shape, device=DEVICE, generator=g)
        else:
            x = x0
    return x[0].cpu().numpy() * st['m_std'] + st['m_mean']

# v2 from mismatched speech (same noise seed as Table 5.2), and from a different speaker's speech
M['v2_mismatched'] = np.stack([sample_from(enc2, den2, test[same_partner[k]], 42 + k) for k in range(N)])
v2_diffspk = np.stack([sample_from(enc2, den2, test[diff_partner[k]], 42 + k) for k in range(N)])
# Five samples per sequence (s = 0..4) for v2 and v1
S = 5
v2_s = np.stack([M['v2']] + [np.stack([sample_v2(it, 42 + k + 1000 * s) for k, it in enumerate(test)]) for s in range(1, S)])
v1_s = np.stack([M['v1']] + [np.stack([sample_v1(it['name'], 42 + k + 1000 * s) for k, it in enumerate(test)]) for s in range(1, S)])
np.savez_compressed(f'{EXTRA}/samples_5_seeds.npz', v1=v1_s, v2=v2_s, v2_diff_speaker=v2_diffspk)
save_motions()

# ---------- Per-sequence measures for every condition ----------
P['real_mismatched'] = per_seq('real', mismatched_names)                  # real motion scored against other speech
P['real_unrelated'] = per_seq('real_unrelated')
P['static'] = per_seq('static')
P['v2_mismatched'] = per_seq('v2_mismatched')                             # generated from other speech, scored vs own

# ---------- FGD (statistical features) and FGD in 20 principal components ----------
feat = {c: np.stack([stat_feat(m) for m in M[c]]) for c in M}
feat['real_mismatched'] = feat['real']                                    # same motions, re-paired -> FGD 0
R = feat['real']
mu_R = R.mean(0); _, _, Vt = np.linalg.svd(R - mu_R, full_matrices=False); PC = Vt[:20].T   # PCA fitted on real
fgd_pca = lambda G: fgd(R @ PC, G @ PC)

# ---------- Speech dependence and diversity ----------
dist = lambda a, b: np.linalg.norm(a[:, UB] - b[:, UB], axis=1).mean()   # mean per-frame Euclidean distance, 63 params
d_real_same = np.array([dist(M['real'][k], M['real'][same_partner[k]]) for k in range(N)])
d_noise     = np.array([dist(v2_s[0][k], v2_s[1][k]) for k in range(N)])
d_diffspk   = np.array([dist(M['v2'][k], v2_diffspk[k]) for k in range(N)])
d_samespk   = np.array([dist(M['v2'][k], M['v2_mismatched'][k]) for k in range(N)])
d_div       = np.array([np.mean([dist(v2_s[a][k], v2_s[b][k]) for a in range(S) for b in range(a + 1, S)]) for k in range(N)])
print('\n===== SPEECH DEPENDENCE (mean per-frame distance, 63 body parameters) =====')
print(f'two real recordings, same speaker : {d_real_same.mean():.2f}')
print(f'v2, noise changed only            : {d_noise.mean():.2f}')
print(f'v2, speech from another speaker   : {d_diffspk.mean():.2f}')
print(f'v2, speech from same speaker      : {d_samespk.mean():.2f}  (below noise in {(d_samespk < d_noise).sum()} of {N})')
print(f'diversity, 5 samples              : {d_div.mean():.2f}  ({100 * d_div.mean() / d_real_same.mean():.0f}% of real same-speaker)')
controls = pd.DataFrame([
    wilcoxon_row('different speaker', 'noise only', 'v2 output distance', d_diffspk, d_noise),
    wilcoxon_row('same speaker', 'noise only', 'v2 output distance', d_samespk, d_noise),
    wilcoxon_row('real own speech', 'real mismatched', 'BeatAlign, improved', re.beatalign_improved, P['real_mismatched'].beatalign_improved),
    wilcoxon_row('real own speech', 'real mismatched', 'BeatAlign, original', re.beatalign_original, P['real_mismatched'].beatalign_original),
    wilcoxon_row('v2 own speech', 'v2 mismatched', 'BeatAlign, improved', v2.beatalign_improved, P['v2_mismatched'].beatalign_improved)])
controls.to_csv(f'{EXTRA}/controls_wilcoxon.csv', index=False)
print(controls.round(4).to_string(index=False))

# ---------- FGD robustness: five seeds and paired bootstrap ----------
def fgd_fast(A, G):
    """Same value as fgd(), computed from the N x N Gram matrix instead of a 660 x 660 matrix square root:
    with N < 660 samples both covariances are low-rank, and tr sqrt(S_A S_G) = sum of sqrt of the eigenvalues
    of (Xa Xg^T)(Xg Xa^T) / ((Na-1)(Ng-1)). About 500x faster, which makes 2,000 bootstrap FGDs practical."""
    Xa, Xg = A - A.mean(0), G - G.mean(0)
    C = Xa @ Xg.T
    ev = np.linalg.eigvalsh(C @ C.T / ((len(A) - 1) * (len(G) - 1)))
    tr = (Xa ** 2).sum() / (len(A) - 1) + (Xg ** 2).sum() / (len(G) - 1)
    return float(((A.mean(0) - G.mean(0)) ** 2).sum() + tr - 2 * np.sqrt(np.clip(ev, 0, None)).sum())
for m in ['v1', 'v2']:                                   # check the shortcut against Cell C's fgd()
    assert abs(fgd_fast(R, feat[m]) - fgd(R, feat[m])) < 1e-3 * max(1.0, fgd(R, feat[m])), m
fgd_seeds = {m: [fgd(R, np.stack([stat_feat(x) for x in arr[s]])) for s in range(S)] for m, arr in [('v1', v1_s), ('v2', v2_s)]}
boot_rng = np.random.RandomState(0); B = 1000; boot = {'v1': [], 'v2': []}
for _ in range(B):
    idx = boot_rng.randint(0, N, N)
    for m in boot:
        boot[m].append(fgd_fast(R[idx], feat[m][idx]))
boot = {m: np.array(v) for m, v in boot.items()}
robust = dict(
    fgd_5_seeds={m: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), values=v) for m, v in fgd_seeds.items()},
    bootstrap_95ci={m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in boot.items()},
    bootstrap_v2_lower_count=int((boot['v2'] < boot['v1']).sum()), bootstrap_B=B,
    diversity=float(d_div.mean()), real_same_speaker=float(d_real_same.mean()))
json.dump(robust, open(f'{EXTRA}/fgd_robustness.json', 'w'), indent=2)
print('\n===== FGD ROBUSTNESS =====')
for m in ['v2', 'v1']:
    print(f"{m}: 5 seeds {robust['fgd_5_seeds'][m]['mean']:.2f} +/- {robust['fgd_5_seeds'][m]['sd']:.2f} | "
          f"bootstrap 95% CI {robust['bootstrap_95ci'][m][0]:.2f}-{robust['bootstrap_95ci'][m][1]:.2f}")
print(f"v2 lower than v1 in {robust['bootstrap_v2_lower_count']} of {B} resamples")

# ---------- Table 5.4 ----------
def table_row(label, cond, fgd_key, ratios_apply=True):
    p = P[cond].drop(columns='clip').mean()
    na = np.nan if not ratios_apply else None
    return {'Condition': label,
            'FGD': np.nan if fgd_key is None else fgd(R, feat[fgd_key]),
            'FGD PCA-20': np.nan if fgd_key is None else fgd_pca(feat[fgd_key]),
            'Jitter ratio': na if na is not None else p.jitter_ratio,
            'Pose spread': na if na is not None else p.pose_std_ratio,
            'Temporal range': na if na is not None else p.temporal_range_ratio,
            'BeatAlign (orig.)': p.beatalign_original, 'BeatAlign (impr.)': p.beatalign_improved}
T54 = [table_row('Real motion', 'real', None),
       table_row('Real motion, mismatched speech', 'real_mismatched', 'real_mismatched', ratios_apply=False),
       table_row('Real motion, unrelated recording', 'real_unrelated', 'real_unrelated'),
       table_row('Static mean pose', 'static', 'static'),
       table_row('v1 (baseline)', 'v1', 'v1'),
       table_row('v2 (improved)', 'v2', 'v2'),
       table_row('v2, mismatched speech', 'v2_mismatched', 'v2_mismatched')]
print('\n===== TABLE 5.4 (without the ablation row, which Cell G adds) =====')
print(pd.DataFrame(T54).round(4).to_string(index=False))


# ================= CELL G: ablation - v2 retrained WITHOUT the velocity loss (~20-40 min on T4) =================
# Needs Cells B, E and F. Everything is identical to Cell B except LAMBDA_VEL = 0 (and, in Cell H, speech dropout).
def train_variant(ckpt_dir, lambda_vel=LAMBDA_VEL, p_uncond=0.0):
    """Cell B's training loop as a function. p_uncond > 0 replaces the speech input (audio and words) with zeros
    for that fraction of training windows, so the network also learns an unconditional prediction (Cell H)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(42); np.random.seed(42)
    enc, den = SpeechEncoderV2().to(DEVICE), MotionDenoiserV2().to(DEVICE)
    prm = list(enc.parameters()) + list(den.parameters())
    optim = torch.optim.AdamW(prm, lr=LR, weight_decay=0.01); scl = torch.amp.GradScaler('cuda')

    def loss_fn(motion, audio, words, gen=None, drop=0.0):
        if drop > 0:
            keep = (torch.rand(motion.shape[0], device=DEVICE) >= drop).float().view(-1, 1, 1)
            audio, words = audio * keep, words * keep
        t = torch.randint(0, STEPS, (motion.shape[0],), device=DEVICE, generator=gen)
        noise = torch.randn(motion.shape, device=DEVICE, generator=gen)
        ab = alphas_cumprod[t].view(-1, 1, 1)
        x_t = ab.sqrt() * motion + (1 - ab).sqrt() * noise
        with torch.amp.autocast('cuda'):
            pred = den(x_t, t, enc(audio, words))
        pred = pred.float()
        loss = nn.functional.mse_loss(pred, motion)
        if lambda_vel > 0:
            loss = loss + lambda_vel * nn.functional.mse_loss(pred.diff(dim=1), motion.diff(dim=1))
        return loss

    def save_ck(path, epoch, best, hist):
        local_path = '/content/' + os.path.basename(ckpt_dir) + '_' + os.path.basename(path)
        torch.save(dict(encoder=enc.state_dict(), denoiser=den.state_dict(), opt=optim.state_dict(), scaler=scl.state_dict(),
                        epoch=epoch, best_val=best, history=hist, stats=stats,
                        config=dict(D=D, HEADS=HEADS, N_ENC=N_ENC, N_DEC=N_DEC, STEPS=STEPS, WIN=WIN, FPS=FPS,
                                    LAMBDA_VEL=lambda_vel, P_UNCOND=p_uncond)), local_path)
        try:
            shutil.copy(local_path, path)
        except Exception as e:
            print('  (warning: could not copy checkpoint to Drive this time:', e, ')')

    first, best, hist = 1, float('inf'), []
    last = f'{ckpt_dir}/latest.pt'
    if os.path.exists(last):
        ck_ = torch.load(last, map_location=DEVICE, weights_only=False)
        enc.load_state_dict(ck_['encoder']); den.load_state_dict(ck_['denoiser'])
        optim.load_state_dict(ck_['opt']); scl.load_state_dict(ck_['scaler'])
        first, best, hist = ck_['epoch'] + 1, ck_['best_val'], ck_['history']
        print('Resuming from epoch', ck_['epoch'])
    for epoch in range(first, EPOCHS + 1):
        enc.train(); den.train(); losses = []; t0 = time.time()
        for motion, audio, words in batches(train, WINDOWS_PER_FILE, True):
            loss = loss_fn(motion, audio, words, drop=p_uncond)
            optim.zero_grad(set_to_none=True)
            scl.scale(loss).backward(); scl.unscale_(optim)
            nn.utils.clip_grad_norm_(prm, 1.0)
            scl.step(optim); scl.update(); losses.append(loss.item())
        enc.eval(); den.eval()
        g = torch.Generator(device=DEVICE).manual_seed(0)
        with torch.no_grad():                                           # conditional validation loss, fixed noise
            val_loss = float(np.mean([loss_fn(m, a, w_, g).item() for m, a, w_ in batches(val, 1, False)]))
        hist.append((epoch, float(np.mean(losses)), val_loss))
        print(f'Epoch {epoch:3d}/{EPOCHS} | train {np.mean(losses):.4f} | val {val_loss:.4f} | {time.time()-t0:.0f}s')
        if val_loss < best:
            best = val_loss; save_ck(f'{ckpt_dir}/best.pt', epoch, best, hist)
        if epoch % 10 == 0 or epoch == EPOCHS:
            save_ck(last, epoch, best, hist)
    print(f'Training complete ({os.path.basename(ckpt_dir)}). Best val loss {best:.4f}')

def load_variant(ckpt_dir):
    ck_ = torch.load(f'{ckpt_dir}/best.pt', map_location=DEVICE, weights_only=False)
    enc, den = SpeechEncoderV2().to(DEVICE), MotionDenoiserV2().to(DEVICE)
    enc.load_state_dict(ck_['encoder']); den.load_state_dict(ck_['denoiser']); enc.eval(); den.eval()
    print(os.path.basename(ckpt_dir), 'best checkpoint: epoch', ck_['epoch'], '| val loss %.4f' % ck_['best_val'])
    return enc, den

train_variant(f'{root}/checkpoints_v2_novel', lambda_vel=0.0)
enc_nv, den_nv = load_variant(f'{root}/checkpoints_v2_novel')
M['v2_novel'] = np.stack([sample_from(enc_nv, den_nv, it, 42 + k) for k, it in enumerate(test)])
feat['v2_novel'] = np.stack([stat_feat(m) for m in M['v2_novel']])
P['v2_novel'] = per_seq('v2_novel'); save_motions()
ablation = pd.DataFrame([
    wilcoxon_row('v2 without velocity loss', 'v2', 'Jitter ratio', P['v2_novel'].jitter_ratio, v2.jitter_ratio),
    wilcoxon_row('v2 without velocity loss', 'v2', 'Temporal range ratio', P['v2_novel'].temporal_range_ratio, v2.temporal_range_ratio)])
ablation.to_csv(f'{EXTRA}/ablation_wilcoxon.csv', index=False)
print(ablation.round(4).to_string(index=False))

T54.append(table_row('v2 without velocity loss', 'v2_novel', 'v2_novel'))
T54 = pd.DataFrame(T54); T54.to_csv(f'{EXTRA}/table_5_4_controls.csv', index=False)
print('\n===== TABLE 5.4 (N = %d, first %d frames) =====' % (N, WIN))
print(T54.round(4).to_string(index=False))


# ================= CELL H: classifier-free guidance model - retrain v2 with 10% speech dropout (~20-40 min on T4) =================
# Needs Cell G (train_variant, load_variant). Same seed (42) and settings as v2; only the speech dropout differs.
P_UNCOND = 0.1
train_variant(f'{root}/checkpoints_v2_cfg', lambda_vel=LAMBDA_VEL, p_uncond=P_UNCOND)
enc_cfg, den_cfg = load_variant(f'{root}/checkpoints_v2_cfg')


# ================= CELL I: learned FGD (BEAT2 evaluator) and choice of guidance weight on VALIDATION (~20-30 min) =================
# The learned FGD uses the pretrained BEAT2 motion autoencoder from PantoMatrix (Liu et al., 2024), which needs
# PantoMatrix's own Python 3.9 environment. setup.sh creates it at /content/py39 and clones emage_evaltools.
import subprocess, json
PM = '/content/PantoMatrix'
if not os.path.exists(f'{PM}/emage_evaltools'):
    subprocess.run(['git', 'clone', 'https://github.com/PantoMatrix/PantoMatrix.git', PM], check=True)
    subprocess.run(['bash', 'setup.sh'], cwd=PM, check=True)                  # ~10 min, run once per session

with open(f'{PM}/learned_fgd.py', 'w') as f:
    f.write('''
import sys, json, numpy as np, torch
sys.path.insert(0, ".")
try:
    from emage_evaltools.mertic import FGD          # module name as spelled in PantoMatrix
except ImportError:
    from emage_evaltools.metric import FGD
import emage_utils.rotation_conversions as rc
dev = "cuda" if torch.cuda.is_available() else "cpu"
job = json.load(open(sys.argv[1])); data = np.load(job["npz"])
evaluator = FGD(download_path="./emage_evaltools/")
def to6d(x):                                     # (T, 165) axis-angle -> (1, T, 330) rotation-6D, as in PantoMatrix
    t = x.shape[0]
    return rc.axis_angle_to_rotation_6d(torch.from_numpy(x).float().reshape(1, t, 55, 3)).reshape(1, t, 330).to(dev)
def lfgd(pred, gt):
    evaluator.reset()
    for p, g in zip(pred, gt):
        evaluator.update(to6d(p), to6d(g))
    return float(evaluator.compute())
out = {}
for key, (p, g) in job["pairs"].items():
    out[key] = lfgd(data[p], data[g])
if job.get("bootstrap"):
    rng = np.random.RandomState(0); n = len(data[job["bootstrap"]["gt"]])
    boot = {p: [] for p in job["bootstrap"]["preds"]}
    for _ in range(job["bootstrap"]["B"]):
        idx = rng.randint(0, n, n)
        for p in boot:
            boot[p].append(lfgd(data[p][idx], data[job["bootstrap"]["gt"]][idx]))
    out["bootstrap"] = boot
json.dump(out, open(job["out"], "w"), indent=2)
''')

def learned_fgd(arrays, pairs, bootstrap=None, tag='job'):
    """arrays: {name: (N, WIN, 165)}; pairs: {label: (pred_name, gt_name)}. Runs in PantoMatrix's py39 environment."""
    npz, job, out = f'/content/{tag}.npz', f'/content/{tag}.json', f'/content/{tag}_out.json'
    np.savez(npz, **{k: v.astype(np.float32) for k, v in arrays.items()})
    json.dump(dict(npz=npz, pairs=pairs, out=out, bootstrap=bootstrap), open(job, 'w'))
    subprocess.run(['/content/py39/bin/python', 'learned_fgd.py', job], cwd=PM, check=True)
    return json.load(open(out))

# ---------- Choose w on the validation set (never the test set) ----------
W_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
V = {'real': np.stack([it['motion'][:WIN] for it in val]),
     'v2': np.stack([sample_from(enc2, den2, it, 42 + k) for k, it in enumerate(val)])}
for w in W_GRID:
    V[f'cfg_w{w}'] = np.stack([sample_from(enc_cfg, den_cfg, it, 42 + k, w=w) for k, it in enumerate(val)])
val_fgd = learned_fgd(V, {c: (c, 'real') for c in V if c != 'real'}, tag='val_cfg')
tr_val = {c: np.mean([temporal_range(V[c][k]) / temporal_range(V['real'][k]) for k in range(len(val))]) for c in V if c != 'real'}
val_table = pd.DataFrame({'learned_FGD': val_fgd, 'temporal_range_ratio': tr_val}); val_table.to_csv(f'{EXTRA}/cfg_validation.csv')
print('===== GUIDANCE WEIGHT ON VALIDATION (N = %d) =====' % len(val)); print(val_table.round(4).to_string())
W_BEST = float(min(W_GRID, key=lambda w: val_fgd[f'cfg_w{w}']))
print('Chosen guidance weight w =', W_BEST)
json.dump(dict(w=W_BEST, grid=W_GRID), open(f'{EXTRA}/cfg_weight.json', 'w'))

# ---------- Guided model on the test set, same seeds as Table 5.2 ----------
M['v2_cfg'] = np.stack([sample_from(enc_cfg, den_cfg, it, 42 + k, w=W_BEST) for k, it in enumerate(test)])
feat['v2_cfg'] = np.stack([stat_feat(m) for m in M['v2_cfg']]); P['v2_cfg'] = per_seq('v2_cfg'); save_motions()


# ================= CELL J: EMAGE on the same test audio, Table 5.5, speaker-2 comparison (~20-30 min) =================
# Needs Cells E-I. EMAGE is run with PantoMatrix's public inference script and audio-only weights (H-Liu1997/emage_audio).
EMAGE_IN, EMAGE_OUT = '/content/emage_audio_in', '/content/emage_motion_out'
os.makedirs(EMAGE_IN, exist_ok=True); os.makedirs(EMAGE_OUT, exist_ok=True)
for n in names_test:
    shutil.copy(hf_hub_download('H-Liu1997/BEAT2', f'beat_english_v2.0.0/wave16k/{n}.wav', repo_type='dataset'), f'{EMAGE_IN}/{n}.wav')
subprocess.run(['/content/py39/bin/python', 'test_emage_audio.py', '--audio_folder', EMAGE_IN, '--save_folder', EMAGE_OUT], cwd=PM, check=True)
M['emage'] = np.stack([fix_len(np.load(f'{EMAGE_OUT}/{n}_output.npz')['poses'], WIN) for n in names_test])
for c in ['emage']:
    feat[c] = np.stack([stat_feat(m) for m in M[c]]); P[c] = per_seq(c)
save_motions()

# ---------- Learned FGD for every condition, half-vs-half reference, and paired bootstrap ----------
half = N // 2
L = dict(M, real_half_a=M['real'][:half], real_half_b=M['real'][half:2 * half])
conds = ['real', 'real_unrelated', 'static', 'v1', 'v2', 'v2_cfg', 'emage']
pairs = {c: (c, 'real') for c in conds}; pairs['real_half_a_vs_b'] = ('real_half_a', 'real_half_b')
lf = learned_fgd(L, pairs, bootstrap=dict(preds=['v2', 'v2_cfg'], gt='real', B=1000), tag='test_learned')
bt = {p: np.array(v) for p, v in lf.pop('bootstrap').items()}
print(f"Learned FGD: real vs itself {lf['real']:.2f} | half vs half {lf['real_half_a_vs_b']:.2f}")
print(f"Bootstrap 95% CI: v2 {np.percentile(bt['v2'], 2.5):.2f}-{np.percentile(bt['v2'], 97.5):.2f} | "
      f"v2-CFG {np.percentile(bt['v2_cfg'], 2.5):.2f}-{np.percentile(bt['v2_cfg'], 97.5):.2f} | "
      f"v2-CFG lower in {(bt['v2_cfg'] < bt['v2']).sum()} of {len(bt['v2'])}")

labels = {'real': 'Real motion', 'real_unrelated': 'Real motion, unrelated recording', 'static': 'Static mean pose',
          'v1': 'v1 (baseline)', 'v2': 'v2 (improved)', 'v2_cfg': f'v2-CFG (guided, w = {W_BEST})', 'emage': 'EMAGE (Liu et al., 2024)'}
T55 = pd.DataFrame([{'Condition': labels[c],
                     'Learned FGD': np.nan if c == 'real' else lf[c],
                     'Statistical FGD': np.nan if c == 'real' else fgd(R, feat[c]),
                     'Jitter ratio': P[c].jitter_ratio.mean(), 'Temporal range ratio': P[c].temporal_range_ratio.mean(),
                     'BeatAlign improved': P[c].beatalign_improved.mean()} for c in conds])
T55.to_csv(f'{EXTRA}/table_5_5_guided_emage.csv', index=False)
print('\n===== TABLE 5.5 (N = %d, first %d frames) =====' % (N, WIN)); print(T55.round(4).to_string(index=False))

# ---------- Speaker 2 only (EMAGE's training speaker) ----------
k2 = [k for k, n in enumerate(names_test) if n.startswith('2_')]
L2 = {c: M[c][k2] for c in ['real', 'v2', 'v2_cfg', 'emage']}
lf2 = learned_fgd(L2, {c: (c, 'real') for c in ['v2', 'v2_cfg', 'emage']}, tag='spk2_learned')
tr2 = {c: P[c].temporal_range_ratio.values[k2] for c in ['v2', 'v2_cfg', 'emage']}
spk2 = pd.DataFrame([
    wilcoxon_row('EMAGE', 'v2', 'Temporal range ratio (speaker 2)', tr2['emage'], tr2['v2']),
    wilcoxon_row('EMAGE', 'v2-CFG', 'Temporal range ratio (speaker 2)', tr2['emage'], tr2['v2_cfg'])])
spk2.to_csv(f'{EXTRA}/speaker2_wilcoxon.csv', index=False)
print(f'\n===== SPEAKER 2 ONLY (N = {len(k2)}) =====')
for c in ['emage', 'v2', 'v2_cfg']:
    print(f'{c:7s} learned FGD {lf2[c]:.2f} | temporal range ratio {tr2[c].mean():.2f}')
print(spk2.round(4).to_string(index=False))
json.dump(dict(learned_fgd_test=lf, learned_fgd_speaker2=lf2, n_speaker2=len(k2), w=W_BEST,
               bootstrap_95ci={p: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for p, v in bt.items()},
               bootstrap_cfg_lower_count=int((bt['v2_cfg'] < bt['v2']).sum())),
          open(f'{EXTRA}/section_5_5_results.json', 'w'), indent=2)
print('Saved all Section 5.3-5.5 outputs to', EXTRA)
