import pennylane.numpy as np
import jax.numpy as jnp

# block below comes from https://pennylane.ai/qml/demos/tutorial_haar_measure/
def qr_haar(N):
    """Generate a Haar-random matrix using the QR decomposition."""
    # Step 1
    A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
    Z = A + 1j * B
    # Step 2
    Q, R = np.linalg.qr(Z)
    # Step 3
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])
    # Step 4
    return np.dot(Q, Lambda)

# Paulis: 
def PI():
    return jnp.array([[1.0, 0.0], [0.0,  1.0]]).astype(jnp.float64) + 1j*jnp.array([[0.0,  0.0], [0.0, 0.0]]).astype(jnp.float64)
def PX():
    return jnp.array([[0.0, 1.0], [1.0,  0.0]]).astype(jnp.float64) + 1j*jnp.array([[0.0,  0.0], [0.0, 0.0]]).astype(jnp.float64)
def PY():
    return jnp.array([[0.0, 0.0], [0.0,  0.0]]).astype(jnp.float64) + 1j*jnp.array([[0.0, -1.0], [1.0, 0.0]]).astype(jnp.float64)
def PZ():
    return jnp.array([[1.0, 0.0], [0.0, -1.0]]).astype(jnp.float64) + 1j*jnp.array([[0.0,  0.0], [0.0, 0.0]]).astype(jnp.float64)


def get_op(op_i):
    if op_i == 0:
        op = PI()
    elif op_i == 1:
        op = PX()
    elif op_i == 2:
        op = PY()
    elif op_i == 3:
        op = PZ()
    return op


def get_op_mat(op):
    for i in range(len(op)):
        if i == 0:
            op_mat = get_op(op[i])
        else:
            next_op = get_op(op[i])
            op_mat_temp = op_mat
            op_mat = jnp.kron(op_mat_temp, next_op)

    return op_mat

def str_to_pauli(P):
    if P == 'X':
        return 1
    if P == 'Y':
        return 2
    if P == 'Z':
        return 3




