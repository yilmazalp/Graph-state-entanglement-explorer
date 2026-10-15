import numpy as np
import random
import math


class Statevector:
    def __init__(self, psi):
        self.psi = np.array(psi, dtype=complex)

        # define projectors
        self.proj_zero = np.array([[1, 0], [0, 0]], dtype=complex)
        self.proj_one = np.array([[0, 0], [0, 1]], dtype=complex)

        # define Paulis
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)


    @staticmethod
    def basis_state(bits):

        zero = np.array([1, 0], dtype=complex)
        one = np.array([0, 1], dtype=complex)

        state = np.array([1], dtype=complex)

        for bit in bits:
            if bit == 0:
                state = np.kron(state, zero)
            elif bit == 1:
                state = np.kron(state, one)
            else:
                raise ValueError("invalid qubit")

        return state

    @staticmethod
    def tensor_product(states):
        result_state = np.array([1], dtype=complex)

        for state in states:
            result_state = np.kron(result_state, state)

        return result_state


    def probabilities(self):
        prob_list = []

        for amplitude in self.psi:
            prob_list.append(np.power(np.abs(amplitude), 2))

        return prob_list


    def apply_1q(self, U, q):

        if len(self.psi) < 2:
            raise ValueError("Expected at least one qubit")

        n = int(round(math.log2(len(self.psi))))

        if 2 ** n != len(self.psi):
            raise ValueError("Statevector length must be a power of two")

        product_state = [self.I] * n

        if not (0 <= q < n):
            raise ValueError("Qubit index out of range")

        product_state[q] = U

        self.psi = self.tensor_product(product_state) @ self.psi

        return self.psi


    def apply_cnot(self, control, target):

        if len(self.psi) < 2:
            raise ValueError("Expected at least one qubit")

        n = int(round(math.log2(len(self.psi))))

        if 2 ** n != len(self.psi):
            raise ValueError("Statevector length must be a power of two")

        if not (0 <= control < n and 0 <= target < n):
            raise ValueError("Qubit index out of range")

        if control == target:
            raise ValueError("Control and target must be different")

        product_states_zero = [self.I] * n
        product_states_one = [self.I] * n

        product_states_zero[control] = self.proj_zero
        product_states_one[control] = self.proj_one

        product_states_zero[target] = self.I
        product_states_one[target] = self.X

        self.psi = (self.tensor_product(product_states_zero) + self.tensor_product(product_states_one)) @ self.psi

        return self.psi


    def apply_cz(self, control, target):

        if len(self.psi) < 2:
            raise ValueError("Expected at least one qubit")

        n = int(round(math.log2(len(self.psi))))

        if 2 ** n != len(self.psi):
            raise ValueError("Statevector length must be a power of two")

        if not (0 <= control < n and 0 <= target < n):
            raise ValueError("Qubit index out of range")

        if control == target:
            raise ValueError("Control and target must be different")


        product_states_zero = [self.I] * n
        product_states_one = [self.I] * n

        product_states_zero[control] = self.proj_zero
        product_states_one[control] = self.proj_one

        product_states_zero[target] = self.I
        product_states_one[target] = self.Z

        self.psi = (self.tensor_product(product_states_zero) + self.tensor_product(product_states_one)) @ self.psi

        return self.psi


    def sample_basis(self, shots, rng):
        # compute the probability of each outcomes after a measurement
        probs = self.probabilities()

        measurements = []
        i = 0
        while i < shots:
            outcome = rng.choice(len(probs), p=probs)
            measurements.append(outcome)
            i += 1

        return measurements

    def fidelity(self, phi):
        return np.power(np.abs(np.vdot(self.psi, phi)), 2)



if __name__ == "__main__":

    # take the state |00>
    state = Statevector(Statevector.basis_state([0,0]))

    # prepare Bell state
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    state.apply_1q(H, 0)
    state.apply_cnot(0, 1)

    print(state.psi)

    bell = (1/np.sqrt(2)) * (state.basis_state([0,0]) + state.basis_state([1,1]))

    # measure the Bell
    rng = np.random.default_rng(42)
    print(state.sample_basis(200, rng))

    # compute the fidelity between the Bell state and itself
    print(state.fidelity(state.psi))

    # compute the fidelity between the Bell state and |00>
    print(state.fidelity(state.basis_state([0,0])))

    # compute the fidelity between the Bell state and |01>
    print(state.fidelity(state.basis_state([0, 1])))

    # compute the fidelity between the Bell state and |10>
    print(state.fidelity(state.basis_state([1, 0])))

    # compute the fidelity between the Bell state and |11>
    print(state.fidelity(state.basis_state([1, 1])))






