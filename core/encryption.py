import hashlib
import secrets
import asyncio

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Encryption:
    def __init__(self, key: str):
        self.key = None
        self.aes_key = None
        self.set_key(key)

    def is_valid(self) -> bool:
        return self.key is not None and self.aes_key is not None

    def set_key(self, key: str):
        self.key = key
        if not self.key:
            return
        self.aes_key = self._derive_aes_key()

    def _derive_aes_key(self) -> bytes:
        sha256 = hashlib.sha256()
        sha256.update(self.key.encode("utf-8"))
        return sha256.digest()

    async def encrypt(self, data: bytes) -> (bytes, bytes):
        nonce = secrets.token_bytes(12)
        aesgcm = AESGCM(self.aes_key)
        encrypted_data = await asyncio.to_thread(aesgcm.encrypt, nonce, data, None)
        return nonce, encrypted_data

    async def decrypt(self, nonce: bytes, encrypted_data: bytes) -> bytes:
        aesgcm = AESGCM(self.aes_key)
        decrypted_data = await asyncio.to_thread(
            aesgcm.decrypt, nonce, encrypted_data, None
        )
        return decrypted_data
