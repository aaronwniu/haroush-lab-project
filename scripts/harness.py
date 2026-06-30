import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

def build_features(session, K, target='decision1'):
    """One session -> (X, y). X = past K choices of both monkeys; y = current target choice.
       Drops first K trials (invalid history)."""
    s = session.reset_index(drop=True)
    d1, d2 = s['decision1'].to_numpy(), s['decision2'].to_numpy()
    yt = s[target].to_numpy()
    X, y = [], []
    for t in range(K, len(s)):
        feats = []
        for lag in range(1, K + 1):
            feats += [d1[t - lag], d2[t - lag]]
        X.append(feats); y.append(yt[t])
    return np.array(X), np.array(y)

def evaluate(make_model, K, sessions, target='decision1'):
    """THE HARNESS. Give it:
         make_model : a function that returns a fresh, untrained model  (e.g. LogisticRegression)
         K          : how many trials of history
         sessions   : list of session dataframes
       Returns a dict of mean/std for accuracy, balanced accuracy, log-loss, baseline,
       using leave-one-session-out cross-validation."""
    accs, bals, losses, bases = [], [], [], []
    for test_i in range(len(sessions)):
        Xtr = np.vstack([build_features(sessions[j], K, target)[0] for j in range(len(sessions)) if j != test_i])
        ytr = np.concatenate([build_features(sessions[j], K, target)[1] for j in range(len(sessions)) if j != test_i])
        Xte, yte = build_features(sessions[test_i], K, target)

        model = make_model()
        model.fit(Xtr, ytr)
        preds, probs = model.predict(Xte), model.predict_proba(Xte)

        accs.append(accuracy_score(yte, preds))
        bals.append(balanced_accuracy_score(yte, preds))
        losses.append(log_loss(yte, probs, labels=[0, 1]))
        majority = round(ytr.mean())
        bases.append((yte == majority).mean())

    return {
        'acc_mean': np.mean(accs),   'acc_std': np.std(accs),
        'bal_mean': np.mean(bals),   'bal_std': np.std(bals),
        'loss_mean': np.mean(losses),'loss_std': np.std(losses),
        'base_mean': np.mean(bases),
        'per_fold_acc': accs,
    }
