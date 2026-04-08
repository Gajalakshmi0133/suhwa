class SuhwaEncryption {
    constructor() {
        this.publicKey = null;
        this.initialized = false;
        this.keyCache = {};
    }

    async initialize() {
        if (this.initialized) return;

        try {
            const response = await fetch('/api/encryption/generate-keys', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`Failed to generate keys: ${response.status}`);
            }

            const data = await response.json();
            if (data.success) {
                this.publicKey = data.public_key;
                this.initialized = true;
                console.log('Encryption initialized successfully');
                return true;
            } else {
                throw new Error(data.error || 'Failed to initialize encryption');
            }
        } catch (error) {
            console.error('Encryption initialization failed:', error);
            return false;
        }
    }

    async getRecipientPublicKey(userId) {
        if (this.keyCache[userId]) {
            return this.keyCache[userId];
        }

        try {
            const response = await fetch(`/api/encryption/get-public-key/${userId}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch public key: ${response.status}`);
            }

            const data = await response.json();
            if (data.success) {
                this.keyCache[userId] = data.public_key;
                return data.public_key;
            } else {
                throw new Error(data.error || 'Failed to fetch public key');
            }
        } catch (error) {
            console.error(`Error fetching public key for user ${userId}:`, error);
            return null;
        }
    }

    async encryptMessage(message, recipientId) {
        if (!this.initialized) {
            console.warn('Encryption not initialized');
            return null;
        }

        try {
            const response = await fetch('/api/encryption/encrypt-message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    recipient_id: recipientId
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to encrypt message: ${response.status}`);
            }

            const data = await response.json();
            if (data.success) {
                return {
                    encrypted: data.encrypted,
                    signature: data.signature,
                    algorithm: data.algorithm
                };
            } else {
                throw new Error(data.error || 'Failed to encrypt message');
            }
        } catch (error) {
            console.error('Message encryption failed:', error);
            return null;
        }
    }

    async decryptMessage(encryptedMessage, signature, senderId) {
        if (!this.initialized) {
            console.warn('Encryption not initialized');
            return null;
        }

        try {
            const response = await fetch('/api/encryption/decrypt-message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    encrypted: encryptedMessage,
                    signature: signature,
                    sender_id: senderId
                })
            });

            if (!response.ok) {
                throw new Error(`Failed to decrypt message: ${response.status}`);
            }

            const data = await response.json();
            if (data.success) {
                return {
                    decrypted: data.decrypted,
                    isValid: data.is_signature_valid
                };
            } else {
                throw new Error(data.error || 'Failed to decrypt message');
            }
        } catch (error) {
            console.error('Message decryption failed:', error);
            return null;
        }
    }

    async encryptForGroup(message, memberIds) {
        if (!this.initialized) {
            console.warn('Encryption not initialized');
            return null;
        }

        const encryptedVersions = {};
        
        for (const memberId of memberIds) {
            const encrypted = await this.encryptMessage(message, memberId);
            if (encrypted) {
                encryptedVersions[memberId] = encrypted;
            }
        }

        return encryptedVersions;
    }
}

const suhwaEncryption = new SuhwaEncryption();

async function initializeEncryption() {
    const success = await suhwaEncryption.initialize();
    if (!success) {
        console.warn('Failed to initialize encryption, falling back to unencrypted mode');
    }
    return success;
}

async function encryptAndSendMessage(message, recipientId, socket, eventName, additionalData = {}) {
    if (!suhwaEncryption.initialized) {
        console.warn('Encryption not available, sending unencrypted message');
        return false;
    }

    try {
        const encrypted = await suhwaEncryption.encryptMessage(message, recipientId);
        if (!encrypted) {
            console.error('Failed to encrypt message');
            return false;
        }

        socket.emit(eventName, {
            ...additionalData,
            encrypted_message: encrypted.encrypted,
            signature: encrypted.signature
        });

        return true;
    } catch (error) {
        console.error('Error in encryptAndSendMessage:', error);
        return false;
    }
}

async function decryptAndDisplayMessage(messageData, currentUserId) {
    if (!messageData.is_encrypted) {
        return messageData.message_text;
    }

    if (!suhwaEncryption.initialized) {
        return '[Encrypted message - encryption not available]';
    }

    try {
        const result = await suhwaEncryption.decryptMessage(
            messageData.encrypted_message,
            messageData.signature,
            messageData.user_id
        );

        if (result && result.decrypted) {
            return result.decrypted;
        } else {
            return '[Failed to decrypt message]';
        }
    } catch (error) {
        console.error('Error decrypting message:', error);
        return '[Failed to decrypt message]';
    }
}

function displayEncryptionStatus(isEnabled) {
    const statusElement = document.getElementById('encryptionStatus');
    if (statusElement) {
        if (isEnabled) {
            statusElement.innerHTML = '<i class="fas fa-lock" style="color: #4CAF50;"></i> Encrypted';
            statusElement.title = 'Messages are encrypted end-to-end';
        } else {
            statusElement.innerHTML = '<i class="fas fa-lock-open" style="color: #FF9800;"></i> Unencrypted';
            statusElement.title = 'Encryption is not available';
        }
    }
}

function addEncryptionBadge(messageElement, isEncrypted) {
    if (isEncrypted) {
        const badge = document.createElement('span');
        badge.className = 'encryption-badge';
        badge.title = 'This message is encrypted end-to-end';
        badge.innerHTML = '<i class="fas fa-lock"></i>';
        badge.style.cssText = `
            display: inline-block;
            font-size: 0.7rem;
            color: #4CAF50;
            margin-left: 0.3rem;
            vertical-align: super;
        `;
        messageElement.appendChild(badge);
    }
}
