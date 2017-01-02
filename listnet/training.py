import logging

import chainer
import chainer.functions as F
import numpy as np


def mean_average_precision(probs, labels, length, th):
    """
    Args:
        probs (numpy.ndarray): list of lists of probability predictions.
            [[0.1, 0.8, 0.3, 0.1, 0.6, 0.6, 0.2], [0., ...]...]
        labels (numpy.ndarray): list of lists of ground-truth labels.
        order (str): {'descending' or 'accending'}

    Return:
        float
    """
    num_queries = len(probs)
    out = 0.0

    for i in xrange(len(probs)):
        r = probs[i][:length[i]].argsort()
        r = r[::-1]
        candidates = labels[i, r]
        avg_prec = 0.
        precisions = []
        num_correct = 0.
        for i in xrange(min(th, len(candidates))):
            if candidates[i] == 1:
                num_correct += 1
                precisions.append(num_correct / (i + 1))

        if precisions:
            avg_prec = sum(precisions) / len(precisions)

            out += avg_prec
    return out / float(num_queries)


def permutation_probability_loss(x, t, length):
    """Calculate permutation probability distributions (k=1) and measure the
    cross entropy over the two distributions.

    Args:
        x (Variable): Variable holding a 2d array whose element
            indicates unnormalized log probability: the first axis of the
            variable represents the number of samples, and the second axis
            represents the number data in a query.
        t (Variable): Variable holding a 2d float32 vector of ground truth
            scores. Must be in same size as x.

    Returns:
        Variable: A variable holding a scalar array of the loss.

    """
    length = length.reshape(-1, 1)
    # log_p: (batch size, n)
    log_p_x = x - F.broadcast_to(F.expand_dims(F.logsumexp(x, axis=1), 1), x.shape)
    # p_t: (batch size, n)
    p_t = F.softmax(t)

    # loss normalized over all instances
    loss = p_t * log_p_x
    mask = np.tile(np.arange(x.shape[1]).reshape(1, -1), (x.shape[0],  1)) < length
    mask = chainer.Variable(mask)
    padding = chainer.Variable(np.zeros(x.shape, dtype=x.dtype))
    loss = F.where(mask, loss, padding)

    return -F.sum(loss / length) / p_t.shape[0]


def clip_data(x, l):
    return x[:, :max(l)]


def _run_batch(model, optimizer, batch, device, train):
    assert train == (optimizer is not None)
    model.cleargrads()

    x, t, l = chainer.dataset.concat_examples(batch, device=device)
    x = clip_data(x, l)
    t = clip_data(t, l)

    y = model(chainer.Variable(x), train=train)
    loss = permutation_probability_loss(y, chainer.Variable(t), l)
    acc = mean_average_precision(y.data, t, l, 100000)
    if optimizer is not None:
        loss.backward()
        optimizer.update()
    return float(loss.data), acc


def forward_pred(model, dataset, device=None):
    loss = 0.
    acc = 0.
    iterator = chainer.iterators.SerialIterator(dataset, batch_size=4,
                                                repeat=False, shuffle=False)
    for batch in iterator:
        l, a = _run_batch(model, None, batch, device, False)
        loss += l * len(batch)
        acc += a * len(batch)
    return loss / float(len(dataset)), acc / float(len(dataset))


def train(model, optimizer, train_itr, n_epoch, dev=None, device=None,
          tmp_dir='tmp.model', lr_decay=0.95):
    loss = 0.
    acc = 0.
    min_loss = float('inf')
    min_epoch = 0
    report_tmpl = "[{:>3d}] T/loss={:0.6f} T/acc={:0.6f} D/loss={:0.6f} D/acc={:0.6f} lr={:0.6f}"
    for batch in train_itr:
        if train_itr.is_new_epoch:
            # this is not executed at first epoch
            loss_dev, acc_dev = forward_pred(model, dev, device=device)
            loss = loss / len(train_itr.dataset)
            acc = acc / len(train_itr.dataset)
            logging.info(report_tmpl.format(
                train_itr.epoch - 1, loss, acc, loss_dev, acc_dev, optimizer.alpha))
            if loss_dev < min_loss:
                min_loss = loss_dev
                min_epoch = train_itr.epoch - 1
                chainer.serializers.save_npz(tmp_dir, model)

            loss = 0.
            acc = 0.
            optimizer.alpha *= lr_decay
        if train_itr.epoch == n_epoch:
            break
        l, a = _run_batch(model, optimizer, batch, device, True)
        loss += l * len(batch)
        acc += a * len(batch)

    logging.info('loading early stopped-model at epoch {}'.format(min_epoch))
    chainer.serializers.load_npz(tmp_dir, model)
