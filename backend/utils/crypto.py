import os
import json
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import hashlib
from typing import Tuple, Dict, Optional


class SignalCrypto:
    """
    End-to-end encryption using Signal Protocol inspired approach.
    Provides key generation, message encryption/decryption for user pairs.
    """
    
    def __init__(self):
        self.backend = default_backend()
    
    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """
        Generate RSA key pair for a user.
        Returns: (public_key_pem, private_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return public_pem, private_pem
    
    @staticmethod
    def encrypt_message(message: str, recipient_public_key: str) -> str:
        """
        Encrypt a message using recipient's public key.
        Args:
            message: Plain text message
            recipient_public_key: PEM formatted public key
        Returns:
            Base64 encoded encrypted message
        """
        public_key = serialization.load_pem_public_key(
            recipient_public_key.encode('utf-8'),
            backend=default_backend()
        )
        
        encrypted = public_key.encrypt(
            message.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    @staticmethod
    def decrypt_message(encrypted_message: str, private_key: str) -> str:
        """
        Decrypt a message using private key.
        Args:
            encrypted_message: Base64 encoded encrypted message
            private_key: PEM formatted private key
        Returns:
            Decrypted plain text message
        """
        priv_key = serialization.load_pem_private_key(
            private_key.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
        
        encrypted_data = base64.b64decode(encrypted_message.encode('utf-8'))
        
        decrypted = priv_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return decrypted.decode('utf-8')
    
    @staticmethod
    def derive_shared_secret(public_key_1: str, private_key_1: str) -> str:
        """
        Derive a shared secret using both users' keys for symmetric encryption.
        Used for group messages and session keys.
        """
        combined = f"{public_key_1}{private_key_1}".encode('utf-8')
        secret = hashlib.sha256(combined).digest()
        return base64.b64encode(secret).decode('utf-8')
    
    @staticmethod
    def encrypt_symmetric(message: str, shared_secret: str) -> str:
        """
        Symmetric encryption using Fernet with shared secret.
        Useful for group messages.
        """
        key = base64.b64encode(
            hashlib.sha256(shared_secret.encode('utf-8')).digest()
        )
        cipher = Fernet(key)
        encrypted = cipher.encrypt(message.encode('utf-8'))
        return encrypted.decode('utf-8')
    
    @staticmethod
    def decrypt_symmetric(encrypted_message: str, shared_secret: str) -> str:
        """
        Decrypt symmetric encrypted message using shared secret.
        """
        key = base64.b64encode(
            hashlib.sha256(shared_secret.encode('utf-8')).digest()
        )
        cipher = Fernet(key)
        decrypted = cipher.decrypt(encrypted_message.encode('utf-8'))
        return decrypted.decode('utf-8')
    
    @staticmethod
    def create_message_signature(message: str, private_key: str) -> str:
        """
        Create a digital signature for message integrity verification.
        """
        priv_key = serialization.load_pem_private_key(
            private_key.encode('utf-8'),
            password=None,
            backend=default_backend()
        )
        
        signature = priv_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode('utf-8')
    
    @staticmethod
    def verify_signature(message: str, signature: str, public_key: str) -> bool:
        """
        Verify message signature using sender's public key.
        """
        try:
            pub_key = serialization.load_pem_public_key(
                public_key.encode('utf-8'),
                backend=default_backend()
            )
            
            sig_bytes = base64.b64decode(signature.encode('utf-8'))
            
            pub_key.verify(
                sig_bytes,
                message.encode('utf-8'),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False


class MessageCryption:
    """
    High-level API for message encryption/decryption in chat contexts.
    Handles both individual and group messages.
    """
    
    def __init__(self, crypto: Optional[SignalCrypto] = None):
        self.crypto = crypto or SignalCrypto()
    
    def encrypt_for_recipient(self, message: str, recipient_public_key: str, 
                            sender_private_key: str) -> Dict:
        """
        Encrypt message for a specific recipient with signature.
        """
        encrypted_text = self.crypto.encrypt_message(message, recipient_public_key)
        signature = self.crypto.create_message_signature(message, sender_private_key)
        
        return {
            'encrypted': encrypted_text,
            'signature': signature,
            'algorithm': 'RSA-OAEP-SHA256'
        }
    
    def decrypt_message(self, encrypted_data: Dict, recipient_private_key: str,
                       sender_public_key: str) -> Tuple[bool, str]:
        """
        Decrypt message and verify sender's signature.
        Returns: (is_valid, decrypted_message)
        """
        try:
            decrypted = self.crypto.decrypt_message(
                encrypted_data['encrypted'],
                recipient_private_key
            )
            
            is_valid = self.crypto.verify_signature(
                decrypted,
                encrypted_data.get('signature', ''),
                sender_public_key
            )
            
            return is_valid, decrypted
        except Exception as e:
            print(f"Decryption failed: {e}")
            return False, ""
    
    def encrypt_for_group(self, message: str, group_members_public_keys: list,
                         sender_private_key: str) -> Dict:
        """
        Encrypt message for group (encrypt separately for each member).
        Returns encrypted versions for each member.
        """
        signature = self.crypto.create_message_signature(message, sender_private_key)
        
        encrypted_versions = {}
        for member_key in group_members_public_keys:
            encrypted_text = self.crypto.encrypt_message(message, member_key)
            encrypted_versions[member_key[:50]] = encrypted_text
        
        return {
            'encrypted_versions': encrypted_versions,
            'signature': signature,
            'algorithm': 'RSA-OAEP-SHA256'
        }


crypto = SignalCrypto()
message_encryption = MessageCryption(crypto)
