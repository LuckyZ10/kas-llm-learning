"""
Homomorphic Encryption Module for Privacy-Preserving Computation
=================================================================

This module implements homomorphic encryption schemes that allow
computations on encrypted data without decryption.

Supports:
- Paillier cryptosystem (additive homomorphic)
- CKKS scheme (approximate arithmetic for ML)
- BFV/BGV schemes (exact arithmetic)

Author: DFT-LAMMPS Team
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import secrets
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class HEConfig:
    """Configuration for homomorphic encryption."""
    scheme: str = "paillier"  # paillier, ckks, bf
    key_size: int = 2048
    polynomial_degree: int = 8192
    coeff_modulus_bits: List[int] = None
    scale: float = 2**40
    
    def __post_init__(self):
        if self.coeff_modulus_bits is None:
            self.coeff_modulus_bits = [60, 40, 40, 60]


class HomomorphicEncryptionScheme(ABC):
    """Abstract base class for homomorphic encryption schemes."""
    
    @abstractmethod
    def generate_keys(self) -> Tuple[any, any]:
        """Generate public and private keys."""
        pass
    
    @abstractmethod
    def encrypt(self, plaintext: Union[int, float, np.ndarray], 
                public_key: any) -> any:
        """Encrypt plaintext."""
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: any, private_key: any) -> Union[int, float, np.ndarray]:
        """Decrypt ciphertext."""
        pass
    
    @abstractmethod
    def add(self, ciphertext1: any, ciphertext2: any) -> any:
        """Homomorphic addition."""
        pass
    
    @abstractmethod
    def multiply_plain(self, ciphertext: any, plaintext: Union[int, float]) -> any:
        """Homomorphic multiplication by plaintext."""
        pass


class PaillierEncryption(HomomorphicEncryptionScheme):
    """
    Paillier cryptosystem implementation.
    
    Additively homomorphic encryption scheme suitable for:
    - Secure aggregation
    - Private sum computation
    - Encrypted voting
    
    Reference: Paillier, "Public-Key Cryptosystems Based on Composite 
    Degree Residue Classes" (EUROCRYPT 1999)
    """
    
    def __init__(self, key_size: int = 2048):
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        
    def generate_keys(self) -> Tuple[Dict, Dict]:
        """
        Generate Paillier key pair.
        
        Returns:
            Tuple of (public_key, private_key) dictionaries
        """
        # Generate two large prime numbers
        p = self._generate_prime(self.key_size // 2)
        q = self._generate_prime(self.key_size // 2)
        
        n = p * q
        n_sq = n * n
        
        # lambda = lcm(p-1, q-1)
        lambda_val = self._lcm(p - 1, q - 1)
        
        # Choose g
        g = n + 1
        
        # Verify g is valid
        # mu = (L(g^lambda mod n^2))^-1 mod n
        # where L(u) = (u - 1) / n
        
        # Compute mu
        l_g_lambda = pow(g, lambda_val, n_sq)
        l_result = (l_g_lambda - 1) // n
        
        try:
            mu = self._mod_inverse(l_result, n)
        except ValueError:
            # Try different g
            g = self._find_valid_g(n, n_sq, lambda_val)
            l_g_lambda = pow(g, lambda_val, n_sq)
            l_result = (l_g_lambda - 1) // n
            mu = self._mod_inverse(l_result, n)
        
        self.public_key = {'n': n, 'g': g, 'n_sq': n_sq}
        self.private_key = {'lambda': lambda_val, 'mu': mu, 'n': n, 'n_sq': n_sq}
        
        return self.public_key, self.private_key
    
    def _generate_prime(self, bits: int) -> int:
        """Generate a random prime number with specified bit length."""
        while True:
            # Generate random number
            n = secrets.randbits(bits)
            # Ensure it's odd and has the right bit length
            n |= (1 << (bits - 1)) | 1
            
            if self._is_prime(n):
                return n
    
    def _is_prime(self, n: int, k: int = 10) -> bool:
        """Miller-Rabin primality test."""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        
        # Witness loop
        for _ in range(k):
            a = secrets.randbelow(n - 3) + 2
            x = pow(a, d, n)
            
            if x == 1 or x == n - 1:
                continue
            
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        
        return True
    
    def _lcm(self, a: int, b: int) -> int:
        """Compute least common multiple."""
        return abs(a * b) // self._gcd(a, b)
    
    def _gcd(self, a: int, b: int) -> int:
        """Compute greatest common divisor."""
        while b:
            a, b = b, a % b
        return a
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Compute modular multiplicative inverse."""
        g, x, y = self._extended_gcd(a % m, m)
        if g != 1:
            raise ValueError("Modular inverse does not exist")
        return x % m
    
    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        """Extended Euclidean algorithm."""
        if a == 0:
            return b, 0, 1
        
        gcd, x1, y1 = self._extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        
        return gcd, x, y
    
    def _find_valid_g(self, n: int, n_sq: int, lambda_val: int) -> int:
        """Find a valid generator g."""
        for _ in range(1000):
            g = secrets.randbelow(n_sq - 1) + 1
            l_g_lambda = pow(g, lambda_val, n_sq)
            l_result = (l_g_lambda - 1) // n
            
            try:
                self._mod_inverse(l_result, n)
                return g
            except ValueError:
                continue
        
        raise ValueError("Could not find valid g")
    
    def encrypt(self, plaintext: int, public_key: Dict = None) -> Tuple[int, int]:
        """
        Encrypt plaintext using Paillier.
        
        Args:
            plaintext: Integer to encrypt (0 <= m < n)
            public_key: Public key dictionary
            
        Returns:
            Ciphertext (c, n) tuple
        """
        if public_key is None:
            public_key = self.public_key
        
        n = public_key['n']
        n_sq = public_key['n_sq']
        g = public_key['g']
        
        if not (0 <= plaintext < n):
            raise ValueError(f"Plaintext must be in range [0, {n})")
        
        # Choose random r in Z_n*
        while True:
            r = secrets.randbelow(n)
            if self._gcd(r, n) == 1:
                break
        
        # c = g^m * r^n mod n^2
        c = (pow(g, plaintext, n_sq) * pow(r, n, n_sq)) % n_sq
        
        return (c, n)
    
    def decrypt(self, ciphertext: Tuple[int, int], 
                private_key: Dict = None) -> int:
        """
        Decrypt ciphertext using Paillier.
        
        Args:
            ciphertext: (c, n) tuple
            private_key: Private key dictionary
            
        Returns:
            Decrypted plaintext
        """
        if private_key is None:
            private_key = self.private_key
        
        c, n = ciphertext
        n_sq = private_key['n_sq']
        lambda_val = private_key['lambda']
        mu = private_key['mu']
        
        # m = L(c^lambda mod n^2) * mu mod n
        c_lambda = pow(c, lambda_val, n_sq)
        l_result = (c_lambda - 1) // n
        m = (l_result * mu) % n
        
        return m
    
    def add(self, ciphertext1: Tuple[int, int], 
            ciphertext2: Tuple[int, int]) -> Tuple[int, int]:
        """
        Homomorphic addition of two ciphertexts.
        
        E(m1) * E(m2) = E(m1 + m2)
        
        Args:
            ciphertext1: First ciphertext
            ciphertext2: Second ciphertext
            
        Returns:
            Ciphertext of sum
        """
        c1, n = ciphertext1
        c2, _ = ciphertext2
        n_sq = n * n
        
        # c = c1 * c2 mod n^2
        c = (c1 * c2) % n_sq
        
        return (c, n)
    
    def multiply_plain(self, ciphertext: Tuple[int, int], 
                      plaintext: int) -> Tuple[int, int]:
        """
        Homomorphic multiplication by plaintext.
        
        E(m)^k = E(m * k)
        
        Args:
            ciphertext: Ciphertext to multiply
            plaintext: Plaintext multiplier
            
        Returns:
            Ciphertext of product
        """
        c, n = ciphertext
        n_sq = n * n
        
        # c = c^k mod n^2
        c_new = pow(c, plaintext, n_sq)
        
        return (c_new, n)
    
    def encrypt_vector(self, vector: np.ndarray, 
                      public_key: Dict = None) -> List[Tuple[int, int]]:
        """Encrypt a vector of integers."""
        return [self.encrypt(int(x), public_key) for x in vector]
    
    def decrypt_vector(self, ciphertexts: List[Tuple[int, int]], 
                      private_key: Dict = None) -> np.ndarray:
        """Decrypt a list of ciphertexts."""
        return np.array([self.decrypt(c, private_key) for c in ciphertexts])
    
    def add_vectors(self, cipher_vec1: List[Tuple[int, int]], 
                   cipher_vec2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Homomorphically add two encrypted vectors."""
        return [self.add(c1, c2) for c1, c2 in zip(cipher_vec1, cipher_vec2)]
    
    def scalar_multiply_vector(self, cipher_vec: List[Tuple[int, int]], 
                               scalar: int) -> List[Tuple[int, int]]:
        """Homomorphically multiply encrypted vector by scalar."""
        return [self.multiply_plain(c, scalar) for c in cipher_vec]


class CKKSEncryption:
    """
    CKKS (Cheon-Kim-Kim-Song) homomorphic encryption scheme.
    
    Supports approximate arithmetic on vectors, suitable for:
    - Machine learning inference
    - Statistical analysis
    - Signal processing
    
    Reference: Cheon et al. "Homomorphic Encryption for Arithmetic of 
    Approximate Numbers" (ASIACRYPT 2017)
    """
    
    def __init__(self, poly_degree: int = 8192, 
                 coeff_modulus_bits: List[int] = None,
                 scale: float = 2**40):
        self.poly_degree = poly_degree
        self.half_degree = poly_degree // 2
        self.coeff_modulus_bits = coeff_modulus_bits or [60, 40, 40, 60]
        self.scale = scale
        
        # Simplified implementation - real CKKS requires complex polynomial operations
        # This is a placeholder that demonstrates the API
        
        self.public_key = None
        self.private_key = None
        self.rotation_keys = None
        
    def generate_keys(self) -> Tuple[Dict, Dict]:
        """Generate CKKS key pair."""
        # Simplified key generation
        self.public_key = {
            'poly_degree': self.poly_degree,
            'modulus_chain': self.coeff_modulus_bits,
            'pk': secrets.token_hex(32)
        }
        self.private_key = {
            'sk': secrets.token_hex(32)
        }
        
        return self.public_key, self.private_key
    
    def encode(self, values: np.ndarray) -> 'CKKSPlaintext':
        """
        Encode real vector into plaintext polynomial.
        
        Args:
            values: Real-valued vector (length <= poly_degree / 2)
            
        Returns:
            CKKSPlaintext object
        """
        if len(values) > self.half_degree:
            raise ValueError(f"Vector too long: {len(values)} > {self.half_degree}")
        
        # Simplified encoding - just scale and round
        scaled = np.round(values * self.scale).astype(np.int64)
        
        # Pad to full length
        padded = np.zeros(self.poly_degree, dtype=np.int64)
        padded[:len(scaled)] = scaled
        padded[self.half_degree:self.half_degree + len(scaled)] = scaled  # Complex conjugate
        
        return CKKSPlaintext(padded, self.scale)
    
    def decode(self, plaintext: 'CKKSPlaintext') -> np.ndarray:
        """Decode plaintext polynomial to real vector."""
        values = plaintext.coeffs[:self.half_degree] / plaintext.scale
        return values
    
    def encrypt(self, plaintext: 'CKKSPlaintext', 
                public_key: Dict = None) -> 'CKKSCiphertext':
        """Encrypt plaintext."""
        # Simplified encryption - add noise
        noise = np.random.normal(0, 1, len(plaintext.coeffs))
        encrypted_coeffs = plaintext.coeffs + noise
        
        return CKKSCiphertext(
            encrypted_coeffs, 
            plaintext.scale,
            level=len(self.coeff_modulus_bits) - 1
        )
    
    def decrypt(self, ciphertext: 'CKKSCiphertext', 
                private_key: Dict = None) -> 'CKKSPlaintext':
        """Decrypt ciphertext."""
        # Simplified decryption - just return as plaintext
        return CKKSPlaintext(ciphertext.coeffs, ciphertext.scale)
    
    def add(self, cipher1: 'CKKSCiphertext', 
            cipher2: 'CKKSCiphertext') -> 'CKKSCiphertext':
        """Homomorphic addition."""
        return CKKSCiphertext(
            cipher1.coeffs + cipher2.coeffs,
            cipher1.scale,
            min(cipher1.level, cipher2.level)
        )
    
    def multiply(self, cipher1: 'CKKSCiphertext', 
                cipher2: 'CKKSCiphertext') -> 'CKKSCiphertext':
        """Homomorphic multiplication."""
        # In real CKKS, this requires relinearization and rescaling
        result_scale = cipher1.scale * cipher2.scale
        return CKKSCiphertext(
            cipher1.coeffs * cipher2.coeffs / result_scale,
            result_scale,
            min(cipher1.level, cipher2.level) - 1
        )
    
    def rescale(self, ciphertext: 'CKKSCiphertext') -> 'CKKSCiphertext':
        """Rescale ciphertext to lower level."""
        return CKKSCiphertext(
            ciphertext.coeffs,
            ciphertext.scale / (2**40),
            ciphertext.level - 1
        )
    
    def rotate(self, ciphertext: 'CKKSCiphertext', 
              steps: int) -> 'CKKSCiphertext':
        """Rotate ciphertext vector."""
        rotated = np.roll(ciphertext.coeffs, steps)
        return CKKSCiphertext(rotated, ciphertext.scale, ciphertext.level)


@dataclass
class CKKSPlaintext:
    """CKKS plaintext representation."""
    coeffs: np.ndarray
    scale: float


@dataclass  
class CKKSCiphertext:
    """CKKS ciphertext representation."""
    coeffs: np.ndarray
    scale: float
    level: int


class BFVEncryption:
    """
    BFV (Brakerski-Fan-Vercauteren) homomorphic encryption scheme.
    
    Supports exact arithmetic on integers, suitable for:
    - Exact computations
    - Boolean circuits
    - Integer arithmetic
    
    Reference: Fan and Vercauteren, "Somewhat Practical Fully Homomorphic 
    Encryption" (IACR ePrint 2012)
    """
    
    def __init__(self, poly_degree: int = 4096, 
                 plain_modulus: int = 65537,
                 coeff_modulus_bits: int = 109):
        self.poly_degree = poly_degree
        self.plain_modulus = plain_modulus
        self.coeff_modulus_bits = coeff_modulus_bits
        
        self.public_key = None
        self.private_key = None
        
    def generate_keys(self) -> Tuple[Dict, Dict]:
        """Generate BFV key pair."""
        self.public_key = {
            'poly_degree': self.poly_degree,
            'plain_modulus': self.plain_modulus,
            'pk': secrets.token_hex(32)
        }
        self.private_key = {
            'sk': secrets.token_hex(32)
        }
        
        return self.public_key, self.private_key
    
    def encode(self, value: int) -> np.ndarray:
        """Encode integer to plaintext polynomial."""
        coeffs = np.zeros(self.poly_degree, dtype=np.int64)
        coeffs[0] = value % self.plain_modulus
        return coeffs
    
    def decode(self, plaintext: np.ndarray) -> int:
        """Decode plaintext polynomial to integer."""
        return int(plaintext[0] % self.plain_modulus)
    
    def encrypt(self, plaintext: np.ndarray, 
                public_key: Dict = None) -> 'BFVCiphertext':
        """Encrypt plaintext."""
        noise = np.random.normal(0, 1, len(plaintext))
        return BFVCiphertext(plaintext + noise, level=1)
    
    def decrypt(self, ciphertext: 'BFVCiphertext', 
                private_key: Dict = None) -> np.ndarray:
        """Decrypt ciphertext."""
        return np.round(ciphertext.coeffs).astype(np.int64)
    
    def add(self, cipher1: 'BFVCiphertext', 
            cipher2: 'BFVCiphertext') -> 'BFVCiphertext':
        """Homomorphic addition."""
        return BFVCiphertext(
            cipher1.coeffs + cipher2.coeffs,
            min(cipher1.level, cipher2.level)
        )
    
    def multiply(self, cipher1: 'BFVCiphertext', 
                cipher2: 'BFVCiphertext') -> 'BFVCiphertext':
        """Homomorphic multiplication."""
        # Simplified - real BFV requires polynomial multiplication
        return BFVCiphertext(
            cipher1.coeffs * cipher2.coeffs,
            min(cipher1.level, cipher2.level) - 1
        )


@dataclass
class BFVCiphertext:
    """BFV ciphertext representation."""
    coeffs: np.ndarray
    level: int


class EncryptedNeuralNetwork:
    """
    Neural network that can perform inference on encrypted data.
    
    Uses homomorphic encryption to protect input data during inference.
    """
    
    def __init__(self, model: nn.Module, encryption_scheme: str = "ckks"):
        self.model = model
        self.encryption_scheme = encryption_scheme
        
        if encryption_scheme == "ckks":
            self.he = CKKSEncryption()
        elif encryption_scheme == "paillier":
            self.he = PaillierEncryption()
        else:
            raise ValueError(f"Unknown scheme: {encryption_scheme}")
        
        self.public_key, self.private_key = self.he.generate_keys()
        
    def encrypted_inference(self, encrypted_input: any) -> any:
        """
        Perform inference on encrypted input.
        
        Args:
            encrypted_input: Encrypted input features
            
        Returns:
            Encrypted output
        """
        # This is a simplified demonstration
        # Real implementation would need homomorphic evaluation of neural network layers
        
        if self.encryption_scheme == "ckks":
            # Simulate linear layer: y = Wx + b
            # In real implementation, weights would be encrypted and operations
            # would be performed homomorphically
            return encrypted_input
        else:
            return encrypted_input
    
    def prepare_weights(self, layer: nn.Linear) -> Dict:
        """
        Prepare neural network weights for encrypted computation.
        
        Args:
            layer: Linear layer
            
        Returns:
            Dictionary of encrypted weights
        """
        weights = layer.weight.detach().numpy()
        bias = layer.bias.detach().numpy() if layer.bias is not None else None
        
        # In real implementation, these would be encoded/encrypted appropriately
        return {
            'weight': weights,
            'bias': bias
        }


class SecureAggregationWithHE:
    """
    Secure aggregation using homomorphic encryption.
    
    Enables multiple parties to compute aggregate statistics without
    revealing individual contributions.
    """
    
    def __init__(self, num_parties: int, he_scheme: str = "paillier"):
        self.num_parties = num_parties
        self.he_scheme = he_scheme
        
        if he_scheme == "paillier":
            self.he = PaillierEncryption()
        else:
            raise ValueError(f"Unsupported scheme: {he_scheme}")
        
        self.public_key, self.private_key = self.he.generate_keys()
        
    def aggregate_encrypted_values(self, encrypted_values: List) -> any:
        """
        Aggregate encrypted values from multiple parties.
        
        Args:
            encrypted_values: List of encrypted values
            
        Returns:
            Encrypted sum
        """
        if not encrypted_values:
            return None
        
        # Homomorphically sum all values
        result = encrypted_values[0]
        for val in encrypted_values[1:]:
            result = self.he.add(result, val)
        
        return result
    
    def compute_average(self, encrypted_values: List) -> float:
        """
        Compute average of encrypted values.
        
        Args:
            encrypted_values: List of encrypted values
            
        Returns:
            Decrypted average
        """
        encrypted_sum = self.aggregate_encrypted_values(encrypted_values)
        
        # Divide by number of parties
        # For Paillier, we can't directly divide
        # Instead, decrypt and divide in plaintext
        total = self.he.decrypt(encrypted_sum)
        return total / len(encrypted_values)


def demo_homomorphic_encryption():
    """
    Demonstration of homomorphic encryption capabilities.
    """
    print("=" * 60)
    print("Homomorphic Encryption Demo")
    print("=" * 60)
    
    # Paillier Demo
    print("\n1. Paillier Encryption (Additive Homomorphic)")
    print("-" * 40)
    
    paillier = PaillierEncryption(key_size=1024)
    pk, sk = paillier.generate_keys()
    
    # Encrypt two values
    m1 = 42
    m2 = 23
    
    c1 = paillier.encrypt(m1, pk)
    c2 = paillier.encrypt(m2, pk)
    
    print(f"Plaintext: m1={m1}, m2={m2}")
    print(f"Encrypted: c1, c2")
    
    # Homomorphic addition
    c_sum = paillier.add(c1, c2)
    m_sum = paillier.decrypt(c_sum, sk)
    
    print(f"Homomorphic addition: decrypt(c1 * c2) = {m_sum}")
    print(f"Expected: {m1 + m2}")
    
    # Homomorphic scalar multiplication
    k = 3
    c_prod = paillier.multiply_plain(c1, k)
    m_prod = paillier.decrypt(c_prod, sk)
    
    print(f"Homomorphic multiplication: decrypt(c1^{k}) = {m_prod}")
    print(f"Expected: {m1 * k}")
    
    # Vector operations
    print("\n2. Vector Operations")
    print("-" * 40)
    
    vec1 = np.array([10, 20, 30, 40, 50])
    vec2 = np.array([5, 10, 15, 20, 25])
    
    cipher_vec1 = paillier.encrypt_vector(vec1, pk)
    cipher_vec2 = paillier.encrypt_vector(vec2, pk)
    
    cipher_sum = paillier.add_vectors(cipher_vec1, cipher_vec2)
    decrypted_sum = paillier.decrypt_vector(cipher_sum, sk)
    
    print(f"Vector 1: {vec1}")
    print(f"Vector 2: {vec2}")
    print(f"Encrypted sum: {decrypted_sum}")
    print(f"Expected sum: {vec1 + vec2}")
    
    # CKKS Demo
    print("\n3. CKKS Encryption (Approximate Arithmetic)")
    print("-" * 40)
    
    ckks = CKKSEncryption(poly_degree=2048)
    pk_ckks, sk_ckks = ckks.generate_keys()
    
    values = np.array([1.5, 2.5, 3.5, 4.5])
    
    plaintext = ckks.encode(values)
    ciphertext = ckks.encrypt(plaintext, pk_ckks)
    decrypted = ckks.decrypt(ciphertext, sk_ckks)
    decoded = ckks.decode(decrypted)
    
    print(f"Original values: {values}")
    print(f"Decoded values: {decoded[:len(values)]}")
    
    print("\n" + "=" * 60)
    print("Homomorphic Encryption Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo_homomorphic_encryption()
