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
