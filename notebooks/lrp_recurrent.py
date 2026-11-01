"""
Layer-wise Relevance Propagation (LRP) for the three RUL architectures used in the paper
(1-D CNN, LSTM + additive attention, hybrid CNN-LSTM-BiLSTM), implemented from scratch
in PyTorch.  Captum's LRP has no propagation rule for ``nn.LSTM``; this module provides one.

Rules (epsilon-LRP, Bach et al. 2015; recurrent rules from Arras et al. 2017):

* Linear / convolutional layer  z = W a + b   -> epsilon rule
      R_a = a * W^T ( R_z / (z + eps*sign(z)) )                   (bias included in z)
* Activation functions (ReLU, tanh)           -> relevance passes through unchanged
* LSTM cell  c_t = f_t*c_{t-1} + i_t*g_t ,  h_t = o_t*tanh(c_t)
      - relevance of c_t is split between the two summands proportionally to their
        contribution (epsilon rule on the sum node),
      - for every product of a gate (sigmoid output i, f, o) and a signal (g, c_{t-1},
        tanh(c_t)) ALL relevance goes to the signal and the gate receives none
        ("signal-take-all" rule),
      - the relevance of g_t is propagated to (x_t, h_{t-1}) through the epsilon rule
        of the linear map that produced g_t.
* Additive attention  ctx = sum_t alpha_t h_t  -> alpha_t treated as constants (gate-like),
  epsilon rule on the weighted sum.

Two independent implementations are provided and are checked against each other
(see ``tests``): an explicit relevance-propagation pass (``*_explicit``) and an
equivalent "modified-gradient" formulation (``lrp``), which is batched and fast:
relevance = x * d y / d x, evaluated on a copy of the forward graph in which
(i) gate activations are detached, (ii) every nonlinearity ``a=f(z)`` is given the
backward multiplier ``f(z)/(z+eps*sign(z))`` and (iii) the epsilon factor
``c/(c+eps*sign(c))`` is applied to the cell state where it re-enters the recurrence.

All computation is done in float64.
"""
import torch
import torch.nn.functional as F

# helpers


def _sgn(z):
    return torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))


def _stab(z, eps):
    return z + eps * _sgn(z)


def _rho_act(z, act, eps):
    """forward: act(z);  backward multiplier: act(z) / (z + eps*sign(z))  (LRP pass-through)"""
    a = act(z)
    ratio = (a / _stab(z, eps)).detach()
    return a.detach() + (z - z.detach()) * ratio


def _grad_scale(c, eps):
    """identity in the forward pass; gradient scaled by c / (c + eps*sign(c))"""
    ratio = (c / _stab(c, eps)).detach()
    return c.detach() + (c - c.detach()) * ratio


def _p(t):
    return t.detach().double()


# modified-gradient implementation (batched)


def _lstm_dir_forward(x, w_ih, w_hh, b_ih, b_hh, eps, reverse=False):
    B, T, _ = x.shape
    H = w_hh.shape[1]
    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    outs = [None] * T
    steps = range(T - 1, -1, -1) if reverse else range(T)
    for t in steps:
        z = x[:, t] @ w_ih.T + h @ w_hh.T + b_ih + b_hh
        zi, zf, zg, zo = z.chunk(4, dim=1)
        i = torch.sigmoid(zi).detach()
        f = torch.sigmoid(zf).detach()
        o = torch.sigmoid(zo).detach()
        g = _rho_act(zg, torch.tanh, eps)
        c = f * _grad_scale(c, eps) + i * g
        h = o * _rho_act(c, torch.tanh, eps)
        outs[t] = h
    return torch.stack(outs, dim=1)


def _lstm_stack_forward(x, lstm, eps):
    out = x
    for l in range(lstm.num_layers):
        fwd = _lstm_dir_forward(out, _p(getattr(lstm, f"weight_ih_l{l}")), _p(getattr(lstm, f"weight_hh_l{l}")),
                                _p(getattr(lstm, f"bias_ih_l{l}")), _p(getattr(lstm, f"bias_hh_l{l}")), eps)
        if lstm.bidirectional:
            bwd = _lstm_dir_forward(out, _p(getattr(lstm, f"weight_ih_l{l}_reverse")),
                                    _p(getattr(lstm, f"weight_hh_l{l}_reverse")),
                                    _p(getattr(lstm, f"bias_ih_l{l}_reverse")),
                                    _p(getattr(lstm, f"bias_hh_l{l}_reverse")), eps, reverse=True)
            out = torch.cat([fwd, bwd], dim=-1)
        else:
            out = fwd
    return out


def _linear(a, lin):
    return a @ _p(lin.weight).T + _p(lin.bias)


def _forward_modified(model, x, eps):
    """Forward pass of the LRP-modified graph; returns the model output (B,)."""
    if hasattr(model, "bilstm"):  # hybrid CNN-LSTM-BiLSTM
        z = x.transpose(1, 2)
        for idx in (0, 3, 6):
            conv = model.cnn[idx]
            z = F.conv1d(z, _p(conv.weight), _p(conv.bias), padding=conv.padding)
            z = _rho_act(z, F.relu, eps)
        z = z.transpose(1, 2)
        z = _lstm_stack_forward(z, model.lstm, eps)
        z = _lstm_stack_forward(z, model.bilstm, eps)
        z = z[:, -1, :]
        z = _rho_act(_linear(z, model.regressor[0]), F.relu, eps)
        return _linear(z, model.regressor[3]).squeeze(-1)
    if hasattr(model, "attention"):  # LSTM + additive attention
        z = _lstm_stack_forward(x, model.lstm, eps)
        alpha = torch.softmax(z @ _p(model.attention.weight).T + _p(model.attention.bias), dim=1).detach()
        ctx = _grad_scale((alpha * z).sum(1), eps)
        return _linear(ctx, model.regressor).squeeze(-1)
    if hasattr(model, "conv1"):  # 1-D CNN
        z = x.transpose(1, 2)
        z = F.conv1d(z, _p(model.conv1.weight), _p(model.conv1.bias), padding=model.conv1.padding)
        z = _rho_act(z, F.relu, eps)
        z = F.conv1d(z, _p(model.conv2.weight), _p(model.conv2.bias), padding=model.conv2.padding)
        z = _rho_act(z, F.relu, eps)
        z = z.flatten(1)
        z = _rho_act(_linear(z, model.fc1), F.relu, eps)
        return _linear(z, model.fc2).squeeze(-1)
    raise ValueError("unsupported model")


def lrp(model, x, eps=1e-6, return_output=False):
    """
    LRP-epsilon relevance map for a batch of windows.

    x : (B, T, F) tensor.   Returns R of the same shape (float64), R[b,t,f] being the
    relevance of sensor f at cycle t for the predicted RUL of window b.
    """
    x = x.detach().double().clone().requires_grad_(True)
    y = _forward_modified(model, x, eps)
    start = (y / _stab(y, eps)).detach()
    grad = torch.autograd.grad(y, x, grad_outputs=start)[0]
    R = (x * grad).detach()
    return (R, y.detach()) if return_output else R


# explicit relevance-propagation implementation (independent check)


def _lstm_dir_cache(x, w_ih, w_hh, b_ih, b_hh, reverse=False):
    B, T, _ = x.shape
    H = w_hh.shape[1]
    h = x.new_zeros(B, H)
    c = x.new_zeros(B, H)
    cache = {k: [None] * T for k in ("z", "i", "f", "g", "o", "c", "c_prev", "h", "h_prev")}
    steps = list(range(T - 1, -1, -1)) if reverse else list(range(T))
    for t in steps:
        z = x[:, t] @ w_ih.T + h @ w_hh.T + b_ih + b_hh
        zi, zf, zg, zo = z.chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(zi), torch.sigmoid(zf), torch.tanh(zg), torch.sigmoid(zo)
        c_prev, h_prev = c, h
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        for k, v in (("z", z), ("i", i), ("f", f), ("g", g), ("o", o), ("c", c), ("c_prev", c_prev),
                     ("h", h), ("h_prev", h_prev)):
            cache[k][t] = v
    cache["steps"] = steps
    cache["H"] = H
    cache["x"] = x
    return cache


def _lstm_dir_backward(cache, w_ih, w_hh, R_h_ext, eps):
    """propagate relevance of the layer outputs R_h_ext (B,T,H) back to the layer input (B,T,I)"""
    x = cache["x"]
    B, T, I = x.shape
    H = cache["H"]
    R_x = torch.zeros_like(x)
    Rh_rec = x.new_zeros(B, H)  # relevance arriving at h_{t-1} from step t
    Rc_next = x.new_zeros(B, H)  # relevance arriving at c_{t-1} from step t (through the forget path)
    for t in reversed(cache["steps"]):
        Rh = R_h_ext[:, t] + Rh_rec
        Rc = Rh + Rc_next  # signal-take-all through o*tanh(c), relevance passes through tanh
        c = cache["c"][t]
        D = _stab(c, eps)
        t1 = cache["f"][t] * cache["c_prev"][t]
        t2 = cache["i"][t] * cache["g"][t]
        Rc_next = t1 / D * Rc  # -> c_{t-1} (signal of f*c_{t-1}); forget gate gets nothing
        Rg = t2 / D * Rc  # -> g_t (signal of i*g); input gate gets nothing
        zg = cache["z"][t][:, 2 * H:3 * H]
        s = Rg / _stab(zg, eps)  # relevance passes through tanh to the pre-activation z_g
        R_x[:, t] = x[:, t] * (s @ w_ih[2 * H:3 * H])
        Rh_rec = cache["h_prev"][t] * (s @ w_hh[2 * H:3 * H])
    return R_x


def _lstm_stack_explicit(x, lstm):
    """forward with caches; returns list of (layer caches) and output"""
    layers = []
    out = x
    for l in range(lstm.num_layers):
        pf = [_p(getattr(lstm, f"{n}_l{l}")) for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
        cf = _lstm_dir_cache(out, *pf)
        entry = {"fwd": (cf, pf[0], pf[1])}
        outs = [torch.stack(cf["h"], dim=1)]
        if lstm.bidirectional:
            pb = [_p(getattr(lstm, f"{n}_l{l}_reverse")) for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
            cb = _lstm_dir_cache(out, *pb, reverse=True)
            entry["bwd"] = (cb, pb[0], pb[1])
            outs.append(torch.stack(cb["h"], dim=1))
        layers.append(entry)
        out = torch.cat(outs, dim=-1)
    return layers, out


def _lstm_stack_backward(layers, R_out, eps):
    R = R_out
    for entry in reversed(layers):
        cf, w_ih, w_hh = entry["fwd"]
        H = cf["H"]
        if "bwd" in entry:
            cb, w_ih_b, w_hh_b = entry["bwd"]
            R_in = _lstm_dir_backward(cf, w_ih, w_hh, R[..., :H], eps) + \
                _lstm_dir_backward(cb, w_ih_b, w_hh_b, R[..., H:], eps)
        else:
            R_in = _lstm_dir_backward(cf, w_ih, w_hh, R, eps)
        R = R_in
    return R


def _lin_back(a, lin, R_z, z, eps):
    s = R_z / _stab(z, eps)
    return a * (s @ _p(lin.weight))


def _conv_back(a, conv, R_z, eps):
    a = a.detach().clone().requires_grad_(True)
    z = F.conv1d(a, _p(conv.weight), _p(conv.bias), padding=conv.padding)
    s = (R_z / _stab(z.detach(), eps))
    g = torch.autograd.grad(z, a, grad_outputs=s)[0]
    return (a * g).detach()


def lrp_explicit(model, x, eps=1e-6):
    """explicit LRP pass for a batch x (B,T,F); returns relevance (B,T,F) in float64"""
    x = x.detach().double()
    with torch.no_grad():
        if hasattr(model, "bilstm"):
            a0 = x.transpose(1, 2)
            acts = [a0]
            zs = []
            a = a0
            for idx in (0, 3, 6):
                conv = model.cnn[idx]
                z = F.conv1d(a, _p(conv.weight), _p(conv.bias), padding=conv.padding)
                zs.append(z)
                a = F.relu(z)
                acts.append(a)
            seq = a.transpose(1, 2)
            l1, o1 = _lstm_stack_explicit(seq, model.lstm)
            l2, o2 = _lstm_stack_explicit(o1, model.bilstm)
            last = o2[:, -1, :]
            z1 = _linear(last, model.regressor[0])
            a1 = F.relu(z1)
            y = _linear(a1, model.regressor[3]).squeeze(-1)
            R_y = y.unsqueeze(-1)
            R_a1 = _lin_back(a1, model.regressor[3], R_y, y.unsqueeze(-1), eps)
            R_last = _lin_back(last, model.regressor[0], R_a1, z1, eps)
            R_o2 = torch.zeros_like(o2)
            R_o2[:, -1, :] = R_last
            R_o1 = _lstm_stack_backward(l2, R_o2, eps)
            R_seq = _lstm_stack_backward(l1, R_o1, eps)
    if hasattr(model, "bilstm"):
        R_a = R_seq.transpose(1, 2)
        for k, idx in reversed(list(enumerate((0, 3, 6)))):
            R_a = _conv_back(acts[k], model.cnn[idx], R_a, eps)
        return R_a.transpose(1, 2)

    with torch.no_grad():
        if hasattr(model, "attention"):
            layers, out = _lstm_stack_explicit(x, model.lstm)
            alpha = torch.softmax(out @ _p(model.attention.weight).T + _p(model.attention.bias), dim=1)
            ctx = (alpha * out).sum(1)
            y = _linear(ctx, model.regressor).squeeze(-1)
            R_ctx = _lin_back(ctx, model.regressor, y.unsqueeze(-1), y.unsqueeze(-1), eps)
            R_out = alpha * out / _stab(ctx, eps).unsqueeze(1) * R_ctx.unsqueeze(1)
            return _lstm_stack_backward(layers, R_out, eps)
    if hasattr(model, "conv1"):
        a0 = x.transpose(1, 2)
        with torch.no_grad():
            z1 = F.conv1d(a0, _p(model.conv1.weight), _p(model.conv1.bias), padding=model.conv1.padding)
            a1 = F.relu(z1)
            z2 = F.conv1d(a1, _p(model.conv2.weight), _p(model.conv2.bias), padding=model.conv2.padding)
            a2 = F.relu(z2)
            flat = a2.flatten(1)
            zf = _linear(flat, model.fc1)
            af = F.relu(zf)
            y = _linear(af, model.fc2).squeeze(-1)
            R_af = _lin_back(af, model.fc2, y.unsqueeze(-1), y.unsqueeze(-1), eps)
            R_flat = _lin_back(flat, model.fc1, R_af, zf, eps)
        R_a2 = R_flat.view_as(a2)
        R_a1 = _conv_back(a1, model.conv2, R_a2, eps)
        R_a0 = _conv_back(a0, model.conv1, R_a1, eps)
        return R_a0.transpose(1, 2)
    raise ValueError("unsupported model")
