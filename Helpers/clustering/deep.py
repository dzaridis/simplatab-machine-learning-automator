"""Deep and neural clustering of tabular data, with a scikit-learn-like interface (``fit``,
``predict``, ``predict_proba``, ``transform``, ``labels_``):

- DEC (Xie et al., 2016): an autoencoder is pretrained, then its encoder and the cluster centres are
  refined together by sharpening the soft assignments (Student-t kernel, KL divergence to a target
  distribution);
- IDEC (Guo et al., 2017): DEC that keeps the reconstruction loss, preserving the local structure;
- DCN (Yang et al., 2017): an autoencoder trained jointly with a k-means loss in its latent space;
- VaDE (Jiang et al., 2017): a variational autoencoder whose prior is a Gaussian mixture, one
  component per cluster;
- SCARF + k-means (Bahri et al., 2022): contrastive self-supervised embeddings (random feature
  corruption from the empirical marginals), clustered with k-means;
- Self-organising map (Kohonen): a grid of prototypes (MiniSom) clustered by Ward linkage.

The saved models store this module by value (cloudpickle): it only imports numpy, torch and
scikit-learn at prediction time (MiniSom is needed for training only).
"""
import math

import numpy as np

ALPHA = 1.0     # degrees of freedom of the Student-t kernel (DEC, IDEC)
TOL = 0.001     # stop when fewer than 0.1% of the samples change cluster between epochs


def _torch():
    import torch
    return torch


def _mlp(dims, last_activation=False):
    from torch import nn
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2 or last_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _student_t(z, centers):
    dist = ((z.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(2)
    q = (1.0 + dist / ALPHA) ** (-(ALPHA + 1) / 2)
    return q / q.sum(1, keepdim=True)


def _target(q):
    weight = q ** 2 / q.sum(0)
    return weight / weight.sum(1, keepdim=True)


class DeepClusterer:
    """An MLP autoencoder (pretrained on the reconstruction of the samples) and a clustering of its
    latent space; the subclasses refine both together."""

    def __init__(self, n_clusters=3, latent_dim=10, hidden=(256, 128), pretrain_epochs=100, epochs=100,
                 batch_size=256, learning_rate=1e-3, seed=0, device=None):
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.hidden = tuple(hidden)
        self.pretrain_epochs = pretrain_epochs
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = device
        self.pretrained_ = False

    # ---- set-up -----------------------------------------------------------------------
    def _device(self):
        torch = _torch()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _build(self, d):
        self.latent_ = min(self.latent_dim, max(2, d))
        self.encoder = _mlp([d, *self.hidden, self.latent_])
        self.decoder = _mlp([self.latent_, *reversed(self.hidden), d])

    def _modules(self):
        return [self.encoder, self.decoder]

    def _encode(self, x):
        return self.encoder(x)

    def _batch(self, n):
        # Enough steps per epoch on small tables: at least ~10 batches when the data allows it
        return int(max(16, min(self.batch_size, n, max(32, n // 10))))

    def _batches(self, n, generator):
        torch = _torch()
        order = torch.randperm(n, generator=generator)
        size = self._batch(n)
        return [order[i:i + size] for i in range(0, n, size) if len(order[i:i + size]) > 1]

    def _tensor(self, X):
        torch = _torch()
        return torch.as_tensor(np.asarray(X, dtype=np.float32), device=self._device())

    def _start(self, X):
        torch = _torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._build(X.shape[1])
        for module in self._modules():
            module.to(self._device())
        self.generator_ = torch.Generator().manual_seed(self.seed)

    # ---- training ---------------------------------------------------------------------
    def pretrain(self, X):
        """Trains the autoencoder to reconstruct the samples."""
        torch = _torch()
        self._start(X)
        Xt = self._tensor(X)
        params = [p for m in self._modules() for p in m.parameters()]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        for module in self._modules():
            module.train()
        for _ in range(self.pretrain_epochs):
            for idx in self._batches(len(Xt), self.generator_):
                xb = Xt[idx.to(Xt.device)]
                loss = ((self.decoder(self._encode(xb)) - xb) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.pretrained_ = True
        return self

    def fit(self, X, y=None):
        if not self.pretrained_:
            self.pretrain(X)
        self._cluster(np.asarray(X, dtype=np.float32))
        self.labels_ = self.predict(X)
        self.to_cpu()
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def _init_centers(self, z):
        from sklearn.cluster import KMeans
        return KMeans(self.n_clusters, n_init=20, random_state=self.seed).fit(z).cluster_centers_

    def _cluster(self, X):
        raise NotImplementedError

    # ---- inference --------------------------------------------------------------------
    def transform(self, X):
        """The latent representation of the samples."""
        torch = _torch()
        for module in self._modules():
            module.eval()
        out = []
        with torch.no_grad():
            Xt = self._tensor(X)
            for i in range(0, len(Xt), 4096):
                out.append(self._encode(Xt[i:i + 4096]).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.latent_), dtype=np.float32)

    def predict_proba(self, X):
        torch = _torch()
        z = torch.as_tensor(self.transform(X))
        centers = torch.as_tensor(np.asarray(self.centers_, dtype=np.float32))
        return _student_t(z, centers).numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def to_cpu(self):
        for module in self._modules():
            module.to("cpu")
        self.device = "cpu"
        return self


class DEC(DeepClusterer):
    reconstruction = 0.0   # IDEC keeps the reconstruction loss

    def _cluster(self, X):
        torch = _torch()
        Xt = self._tensor(X)
        centers = torch.nn.Parameter(torch.as_tensor(self._init_centers(self.transform(X)), dtype=torch.float32,
                                                     device=Xt.device))
        params = [p for p in self.encoder.parameters()] + [centers]
        if self.reconstruction:
            params += list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        previous = None
        for _ in range(self.epochs):
            self.encoder.eval()
            with torch.no_grad():
                q = _student_t(self._encode(Xt), centers)
                p = _target(q)
                labels = q.argmax(1)
            if previous is not None and (labels != previous).float().mean().item() < TOL:
                break
            previous = labels
            self.encoder.train()
            for idx in self._batches(len(Xt), self.generator_):
                idx = idx.to(Xt.device)
                xb = Xt[idx]
                z = self._encode(xb)
                q_b = _student_t(z, centers)
                loss = (p[idx] * (torch.log(p[idx] + 1e-10) - torch.log(q_b + 1e-10))).sum(1).mean()
                if self.reconstruction:
                    loss = ((self.decoder(z) - xb) ** 2).mean() + self.reconstruction * loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.centers_ = centers.detach().cpu().numpy()


class IDEC(DEC):
    reconstruction = 0.1   # weight of the clustering loss next to the reconstruction (gamma of the paper)


class DCN(DeepClusterer):
    lam = 1.0   # weight of the k-means loss

    def _cluster(self, X):
        torch = _torch()
        Xt = self._tensor(X)
        centers = torch.as_tensor(self._init_centers(self.transform(X)), dtype=torch.float32, device=Xt.device)
        counts = torch.full((self.n_clusters,), 100.0, device=Xt.device)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        previous = None
        for _ in range(self.epochs):
            for module in self._modules():
                module.train()
            for idx in self._batches(len(Xt), self.generator_):
                xb = Xt[idx.to(Xt.device)]
                z = self._encode(xb)
                with torch.no_grad():
                    assign = torch.cdist(z, centers).argmin(1)
                loss = ((self.decoder(z) - xb) ** 2).mean() + \
                    self.lam / 2 * ((z - centers[assign]) ** 2).sum(1).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():   # incremental update of the centres (running means)
                    z = self._encode(xb)
                    for j in assign.unique():
                        members = z[assign == j]
                        centers[j] = (counts[j] * centers[j] + members.sum(0)) / (counts[j] + len(members))
                        counts[j] += len(members)
            with torch.no_grad():
                labels = torch.cdist(self._encode(Xt), centers).argmin(1)
            if previous is not None and (labels != previous).float().mean().item() < TOL:
                break
            previous = labels
        self.centers_ = centers.cpu().numpy()


class VaDE(DeepClusterer):
    """Variational deep embedding: q(z|x) Gaussian, p(z) a mixture of Gaussians (one per cluster),
    p(x|z) Gaussian with unit variance (standardised features)."""

    def _build(self, d):
        from torch import nn
        self.latent_ = min(self.latent_dim, max(2, d))
        self.body = _mlp([d, *self.hidden], last_activation=True)
        self.mu_head = nn.Linear(self.hidden[-1], self.latent_)
        self.logvar_head = nn.Linear(self.hidden[-1], self.latent_)
        self.decoder = _mlp([self.latent_, *reversed(self.hidden), d])

    def _modules(self):
        return [self.body, self.mu_head, self.logvar_head, self.decoder]

    def _encode(self, x):
        return self.mu_head(self.body(x))

    def _log_pzc(self, z, mu_c, logvar_c):
        return -0.5 * (math.log(2 * math.pi) + logvar_c.unsqueeze(0)
                       + (z.unsqueeze(1) - mu_c.unsqueeze(0)) ** 2 / logvar_c.exp().unsqueeze(0)).sum(2)

    def _cluster(self, X):
        torch = _torch()
        from sklearn.mixture import GaussianMixture
        Xt = self._tensor(X)
        z0 = self.transform(X)
        gmm = GaussianMixture(self.n_clusters, covariance_type="diag", n_init=3, reg_covar=1e-4,
                              random_state=self.seed).fit(z0)
        dev = Xt.device
        pi = torch.nn.Parameter(torch.as_tensor(np.log(gmm.weights_ + 1e-6), dtype=torch.float32, device=dev))
        mu_c = torch.nn.Parameter(torch.as_tensor(gmm.means_, dtype=torch.float32, device=dev))
        logvar_c = torch.nn.Parameter(torch.as_tensor(np.log(gmm.covariances_), dtype=torch.float32, device=dev))
        # The posterior variance starts small, as the pretrained (deterministic) encoder
        with torch.no_grad():
            self.logvar_head.weight.mul_(0.01)
            self.logvar_head.bias.fill_(float(np.log(np.mean(gmm.covariances_)) - 2.0))
        params = [p for m in self._modules() for p in m.parameters()] + [pi, mu_c, logvar_c]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        previous = None
        for _ in range(self.epochs):
            for module in self._modules():
                module.train()
            for idx in self._batches(len(Xt), self.generator_):
                xb = Xt[idx.to(dev)]
                h = self.body(xb)
                mu, logvar = self.mu_head(h), self.logvar_head(h).clamp(-12, 8)
                z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                rec = 0.5 * ((self.decoder(z) - xb) ** 2).sum(1)
                lv_c = logvar_c.clamp(-12, 8)
                log_pi = torch.log_softmax(pi, 0)
                gamma = torch.softmax(log_pi + self._log_pzc(z, mu_c, lv_c), 1)
                var_c = lv_c.exp().unsqueeze(0)
                kl_z = 0.5 * (gamma * (lv_c.unsqueeze(0) + logvar.exp().unsqueeze(1) / var_c
                                       + (mu.unsqueeze(1) - mu_c.unsqueeze(0)) ** 2 / var_c).sum(2)).sum(1)
                kl_c = (gamma * (torch.log(gamma + 1e-10) - log_pi)).sum(1)
                entropy = -0.5 * (1 + logvar).sum(1)
                loss = (rec + kl_z + kl_c + entropy).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.pi_, self.mu_c_, self.logvar_c_ = (t.detach().cpu().numpy() for t in (pi, mu_c, logvar_c.clamp(-12, 8)))
            labels = self.predict_proba(X).argmax(1)
            if previous is not None and np.mean(labels != previous) < TOL:
                break
            previous = labels
        self.centers_ = self.mu_c_

    def predict_proba(self, X):
        torch = _torch()
        z = torch.as_tensor(self.transform(X))
        log_pi = torch.log_softmax(torch.as_tensor(self.pi_), 0)
        log_p = log_pi + self._log_pzc(z, torch.as_tensor(self.mu_c_), torch.as_tensor(self.logvar_c_))
        return torch.softmax(log_p, 1).numpy()


class SCARF(DeepClusterer):
    """Contrastive embeddings: each sample and a corrupted view of it (a random 60% of its features
    replaced by values of other samples) are pulled together, other samples pushed apart (InfoNCE).
    The normalised embeddings are clustered with k-means."""

    corruption = 0.6
    temperature = 0.5

    def _build(self, d):
        self.latent_ = max(2, self.latent_dim)
        self.encoder = _mlp([d, *self.hidden, self.latent_])
        self.head = _mlp([self.latent_, self.latent_, self.latent_])

    def _modules(self):
        return [self.encoder, self.head]

    def _encode(self, x):
        import torch.nn.functional as F
        return F.normalize(self.encoder(x), dim=1)

    def pretrain(self, X):
        torch = _torch()
        import torch.nn.functional as F
        self._start(X)
        Xt = self._tensor(X)
        n, d = Xt.shape
        params = [p for m in self._modules() for p in m.parameters()]
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        columns = torch.arange(d, device=Xt.device)
        for module in self._modules():
            module.train()
        for _ in range(self.pretrain_epochs):
            for idx in self._batches(n, self.generator_):
                xb = Xt[idx.to(Xt.device)]
                b = len(xb)
                mask = torch.rand(b, d, generator=self.generator_).to(Xt.device) < self.corruption
                rows = torch.randint(0, n, (b, d), generator=self.generator_).to(Xt.device)
                corrupted = torch.where(mask, Xt[rows, columns], xb)
                z1 = F.normalize(self.head(self.encoder(xb)), dim=1)
                z2 = F.normalize(self.head(self.encoder(corrupted)), dim=1)
                logits = z1 @ z2.T / self.temperature
                target = torch.arange(b, device=Xt.device)
                loss = (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.pretrained_ = True
        return self

    def _cluster(self, X):
        from sklearn.cluster import KMeans
        self.kmeans_ = KMeans(self.n_clusters, n_init=20, random_state=self.seed).fit(self.transform(X))
        self.centers_ = self.kmeans_.cluster_centers_

    def predict(self, X):
        return self.kmeans_.predict(self.transform(X))

    def predict_proba(self, X):
        torch = _torch()
        return _student_t(torch.as_tensor(self.transform(X)), torch.as_tensor(self.centers_, dtype=torch.float32)).numpy()


class SelfOrganizingMap:
    """A Kohonen map: a square grid of prototypes (about 5 sqrt(n) of them) trained on the samples,
    then the prototypes that represent samples are grouped into clusters with Ward linkage; a
    sample belongs to the cluster of its best-matching prototype."""

    def __init__(self, n_clusters=3, seed=0, iterations=None):
        self.n_clusters = n_clusters
        self.seed = seed
        self.iterations = iterations
        self.pretrained_ = True

    def pretrain(self, X):
        return self

    def fit(self, X, y=None):
        from minisom import MiniSom
        from sklearn.cluster import AgglomerativeClustering
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        side = int(min(30, max(3, round(math.sqrt(5 * math.sqrt(n))))))
        som = MiniSom(side, side, d, sigma=max(1.0, side / 4), learning_rate=0.5, random_seed=self.seed)
        if d >= 2:
            som.pca_weights_init(X)
        else:
            som.random_weights_init(X)
        som.train(X, int(self.iterations or max(1000, min(20000, 50 * n))), random_order=True)
        self.codebook_ = som.get_weights().reshape(-1, d)
        winners = self._winners(X)
        hits = np.unique(winners)
        k = max(1, min(self.n_clusters, len(hits)))
        labels = np.zeros(len(self.codebook_), dtype=int)
        if k > 1:
            labels[hits] = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(self.codebook_[hits])
        # Prototypes without samples take the cluster of their nearest prototype with samples
        empty = np.setdiff1d(np.arange(len(self.codebook_)), hits)
        if len(empty):
            nearest = ((self.codebook_[empty][:, None] - self.codebook_[hits][None]) ** 2).sum(2).argmin(1)
            labels[empty] = labels[hits][nearest]
        self.prototype_labels_ = labels
        self.labels_ = labels[winners]
        return self

    def _winners(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for i in range(0, len(X), 4096):
            block = X[i:i + 4096]
            dist = (block ** 2).sum(1)[:, None] - 2 * block @ self.codebook_.T + (self.codebook_ ** 2).sum(1)[None]
            out.append(dist.argmin(1))
        return np.concatenate(out) if out else np.zeros(0, dtype=int)

    def predict(self, X):
        return self.prototype_labels_[self._winners(X)]

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def to_cpu(self):
        return self
