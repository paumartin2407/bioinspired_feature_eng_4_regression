import numpy as np

def candidate_frame(individual, X, kind, reference=None, mode="combined"):
    """Apply a GA/ES/PSO individual to a frame with the same input columns."""
    mask = np.asarray(individual["mask"]).astype(bool)
    if not mask.any():
        mask[0] = True
    out = X.copy() if mode == "construction" else X.loc[:, mask].copy()
    if mode == "selection":
        return out
    for i, coef in enumerate(individual["coefs"]):
        z = X.values @ coef
        if individual["type_mask"][i]:
            z = z ** 2
        if kind == "ES":
            ref = X if reference is None else reference
            z_ref = ref.values @ coef
            if individual["type_mask"][i]:
                z_ref = z_ref ** 2
            z = (z - z_ref.mean()) / (z_ref.std() + 1e-6)
        out[f"synth_{i}"] = z
    return out

def format_individual(individual, feature_names):
    mask = np.asarray(individual["mask"]).astype(bool)
    selected = [name for name, keep in zip(feature_names, mask) if keep]
    formulas = []
    for i, weights in enumerate(individual["coefs"]):
        terms = [f"{weight:+.3f}*{name}" for weight, name in zip(weights, feature_names) if abs(weight) >= 0.05]
        expr = " ".join(terms) if terms else "0"
        if individual["type_mask"][i]:
            expr = f"({expr})^2"
        formulas.append(f"synth_{i}={expr}")
    return "selected=" + ",".join(selected) + "; " + "; ".join(formulas)

def add_gp_feature(tree, X_train, X_val, X_test, name="GP_feature"):
    z_train = tree.evaluate(X_train.values)
    mean, std = z_train.mean(), z_train.std()
    if std < 1e-9:
        z_train = np.zeros_like(z_train)
        z_val = np.zeros(len(X_val))
        z_test = np.zeros(len(X_test))
    else:
        z_train = (z_train - mean) / std
        z_val = (tree.evaluate(X_val.values) - mean) / std
        z_test = (tree.evaluate(X_test.values) - mean) / std
    frames = [frame.copy() for frame in (X_train, X_val, X_test)]
    for frame, z in zip(frames, (z_train, z_val, z_test)):
        frame[name] = z
    return frames

