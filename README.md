# QHack 2024 Open Hackathon Project

## Optimizing the depth of the Mottonen state preparation circuit in PennyLane

Team name: team-penguin

Team Member: Kazuki Tsuoka

## Abstract

My team worked on an exact method for amplitude embedding. When gate operations can be performed simultaneously on different qubits, the execution speed of a quantum circuit depends on the depth of the circuit. Compared to the \codeword{qml.MottonenStatePreparation} circuit implemented in PennyLane, we reduced the depth of the amplitude embedding circuit by $\simeq$ 25\% for >10 qubits by optimizing the placement of the uniformly controlled Pauli-Z rotation (UCRZ) gates based on [Zhang et al.'s paper](https://arxiv.org/abs/2212.01002).
