import numpy as np
import pennylane as qml


def gray_code(rank):
    """Generates the Gray code of given rank.

    Args:
        rank (int): rank of the Gray code (i.e. number of bits)
    """

    def gray_code_recurse(g, rank):
        k = len(g)
        if rank <= 0:
            return

        for i in range(k - 1, -1, -1):
            char = "1" + g[i]
            g.append(char)
        for i in range(k - 1, -1, -1):
            g[i] = "0" + g[i]

        gray_code_recurse(g, rank - 1)

    g = ["0", "1"]
    gray_code_recurse(g, rank - 1)

    return g


def _matrix_M_entry(row, col):
    # (col >> 1) ^ col is the Gray code of col
    b_and_g = row & ((col >> 1) ^ col)
    sum_of_ones = 0
    while b_and_g > 0:
        if b_and_g & 0b1:
            sum_of_ones += 1

        b_and_g = b_and_g >> 1

    return (-1) ** sum_of_ones


def compute_theta(alpha):
    ln = alpha.shape[-1]
    k = np.log2(ln)

    M_trans = np.zeros(shape=(ln, ln))
    for i in range(len(M_trans)):
        for j in range(len(M_trans[0])):
            M_trans[i, j] = _matrix_M_entry(j, i)

    theta = qml.math.transpose(qml.math.dot(M_trans, qml.math.transpose(alpha)))

    return theta / 2**k


def _apply_uniform_rotation_dagger(gate, alpha, control_wires, target_wire, flip):
    theta = compute_theta(alpha)

    gray_code_rank = len(control_wires)

    if gray_code_rank == 0:
        if qml.math.is_abstract(theta) or qml.math.all(theta[0] != 0.0):
            gate(theta[0], wires=[target_wire])
        return None

    code = gray_code(gray_code_rank)
    num_selections = len(code)

    control_indices = [
        int(np.log2(int(code[i], 2) ^ int(code[(i + 1) % num_selections], 2)))
        for i in range(num_selections)
    ]

    if not flip:
        for i, control_index in enumerate(control_indices):
            gate(theta[i], wires=[target_wire])
            if not (i == len(control_indices) - 1 and gate.__name__ == "RY"):
                qml.CNOT(wires=[control_wires[control_index], target_wire])
    else:
        control_indices.reverse()
        for i, control_index in enumerate(control_indices):
            if not (i == 0 and gate.__name__ == "RZ"):
                qml.CNOT(wires=[control_wires[control_index], target_wire])
            gate(theta[-i - 1], wires=[target_wire])


def _get_alpha_z(omega, n, k):
    indices1 = [
        [(2 * j - 1) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]
    indices2 = [
        [(2 * j - 2) * 2 ** (k - 1) + l - 1 for l in range(1, 2 ** (k - 1) + 1)]
        for j in range(1, 2 ** (n - k) + 1)
    ]

    term1 = qml.math.take(omega, indices=indices1, axis=-1)
    term2 = qml.math.take(omega, indices=indices2, axis=-1)
    diff = (term1 - term2) / 2 ** (k - 1)

    return qml.math.sum(diff, axis=-1)


def _get_alpha_y(a, n, k):
    indices_numerator = [
        [(2 * (j + 1) - 1) * 2 ** (k - 1) + l for l in range(2 ** (k - 1))]
        for j in range(2 ** (n - k))
    ]
    numerator = qml.math.take(a, indices=indices_numerator, axis=-1)
    numerator = qml.math.sum(qml.math.abs(numerator) ** 2, axis=-1)

    indices_denominator = [
        [j * 2**k + l for l in range(2**k)] for j in range(2 ** (n - k))
    ]
    denominator = qml.math.take(a, indices=indices_denominator, axis=-1)
    denominator = qml.math.sum(qml.math.abs(denominator) ** 2, axis=-1)

    # Divide only where denominator is zero, else leave initial value of zero.
    # The equation guarantees that the numerator is also zero in the corresponding entries.

    with np.errstate(divide="ignore", invalid="ignore"):
        division = numerator / denominator

    # Cast the numerator and denominator to ensure compatibility with interfaces
    division = qml.math.cast(division, np.float64)
    denominator = qml.math.cast(denominator, np.float64)

    division = qml.math.where(denominator != 0.0, division, 0.0)

    return 2 * qml.math.arcsin(qml.math.sqrt(division))


def amplitude_embedding(state_vector, wires):
    # Check that the state vector is valid
    shape = qml.math.shape(state_vector)

    if len(shape) != 1:
        raise ValueError(
            f"State vectors must be one-dimensional; vector has shape {shape}."
        )

    n_amplitudes = shape[0]
    if n_amplitudes != 2 ** len(qml.wires.Wires(wires)):
        raise ValueError(
            f"State vectors must be of length {2 ** len(wires)} or less; vector has length {n_amplitudes}."
        )

    if not qml.math.is_abstract(state_vector):
        norm = qml.math.sum(qml.math.abs(state_vector) ** 2)
        if not qml.math.allclose(norm, 1.0, atol=1e-3):
            raise ValueError(
                f"State vectors have to be of norm 1.0, vector has norm {norm}"
            )

    # Implement the amplitude embedding
    a = qml.math.abs(state_vector)
    omega = qml.math.angle(state_vector)
    # change ordering of wires, since original code
    # was written for IBM machines
    wires_reverse = wires[::-1]

    # Apply inverse y rotation cascade to prepare correct absolute values of amplitudes
    for k in range(len(wires_reverse), 0, -1):
        control = wires_reverse[k:]
        target = wires_reverse[k - 1]

        alpha_y_k = _get_alpha_y(a, len(wires_reverse), k)
        _apply_uniform_rotation_dagger(qml.RY, alpha_y_k, control, target, False)

        alpha_z_k = _get_alpha_z(omega, len(wires_reverse), k)
        _apply_uniform_rotation_dagger(qml.RZ, alpha_z_k, control, target, True)
