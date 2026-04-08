# 🔐 How Suhwa Keeps Your Messages Secure

Welcome to the Suhwa Community! We use **end-to-end encryption** to protect your privacy. Here's how it works:

---

## 1️⃣ **Message Creation** (Sender Side)

You type a message and tap **Send**.

- The message is converted into **digital data** (packets)
- Before leaving your device, it is **encrypted using the Signal Protocol**
- 👉 This encryption means:
  - ✅ **Only you and the receiver** can read the message
  - ✅ **Not even Suhwa servers** can see the content
  - ✅ **No one in between** can intercept and understand your message

---

## 2️⃣ **Encryption Process** 🔐

Suhwa uses **Signal Protocol** for encryption:

### How It Works:

- **Public Key Encryption**: Your device has a unique public key (shared with others)
- **Private Key Encryption**: Your device has a secret private key (known only to you)
- **Unique Session Keys**: Each chat has its own encryption key

### When You Send a Message:

1. Your app **encrypts** the message using the **receiver's public key**
2. **Only the receiver's private key** can decrypt it
3. Even if someone **intercepts the encrypted message**, it looks like random code—completely unreadable

---

## 3️⃣ **Sending to Suhwa Server** 📤

The encrypted message travels via the internet (WiFi / Mobile Data):

1. It reaches **Suhwa's server**
2. The server **DOES NOT decrypt** it (servers can't see your messages!)
3. The server **only forwards** the encrypted data to the receiver
4. **If the receiver is offline:**
   - The server temporarily **stores the encrypted message**
   - Once they come online, it **delivers the message immediately**

---

## 4️⃣ **Delivery to Receiver** 📩

The receiver's device receives the encrypted data:

1. Their app **decrypts** it using their **private key**
2. The **original message appears** in the chat
3. Only they can read what you wrote

---

## 5️⃣ **Blue Tick System** ✔✔

Suhwa also sends small status updates (separate from your actual message):

- **✔ One grey tick** → Message sent to Suhwa server
- **✔✔ Two grey ticks** → Delivered to receiver's device
- **✔✔ Blue ticks** → Read by receiver

These status updates are just small system messages—your actual chat content remains fully encrypted.

---

## 6️⃣ **How Media Works** 🎥 (Photos, Videos, Voice Messages)

Media files are handled securely:

1. **Media file is encrypted** on your device
2. **Uploaded to Suhwa server** (still encrypted)
3. **Receiver downloads** the encrypted file
4. **Decrypted on their device** (only then can they view/listen)

This means:
- ✅ Your photos and videos are **never visible to Suhwa servers**
- ✅ **Only the recipient** can view your media
- ✅ Your **privacy is protected** throughout the entire process

---

## 7️⃣ **Group Messages** 👥

In **group chats**, Suhwa handles encryption for each member:

1. Your message is **encrypted separately for each group member**
2. **Multiple encrypted copies** are sent (one for each member)
3. **Only group members can decrypt** their copy
4. **Non-members cannot access** the group's messages

---

## 8️⃣ **Voice & Video Calls** 📞

Suhwa secures your calls with:

- **End-to-end encrypted calling** (same Signal Protocol)
- **Internet-based voice transmission** (VoIP)
- **Secure key exchange** before the call starts
- Neither Suhwa servers nor anyone listening can understand your conversation

---

## 🧠 **Simple Technical Flow**

```
┌─────────────┐
│   Sender    │  Types a message
└─────┬───────┘
      │
      ▼
┌─────────────────────────────┐
│  ENCRYPT Message            │  Using Signal Protocol
│  (Receiver's Public Key)    │  + Unique Session Key
└─────┬───────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│  Send Encrypted Data via     │  WiFi / Mobile Data
│  Internet to Suhwa Server    │
└─────┬────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│  Suhwa Server                │  Can't read it!
│  (Forwards to Receiver)      │  Just passes it along
└─────┬────────────────────────┘
      │
      ▼
┌──────────────────────────────┐
│  Receiver's Device           │  Gets encrypted message
│  DECRYPT Message             │  (Using Private Key)
│  (Receiver's Private Key)    │
└─────┬────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│   Receiver Reads Message    │  Original text displayed
│   (Readable & Decrypted)    │  Only they can see it
└─────────────────────────────┘
```

---

## ✨ **Key Takeaways**

| Feature | How It Works |
|---------|-------------|
| **Encryption** | End-to-end using Signal Protocol |
| **Who can read?** | Only sender and receiver |
| **What about Suhwa?** | Servers never see message content |
| **Offline messages?** | Stored encrypted, delivered when online |
| **Group chats?** | Encrypted separately for each member |
| **Media files?** | Encrypted during upload & download |
| **Calls?** | Fully encrypted from start to end |

---

## 🛡️ **Your Privacy is Our Priority**

Suhwa is built with privacy at its core. Every message, photo, video, and call is protected by military-grade encryption. We believe your conversations belong to **you and the people you're talking to**—no one else.

**Questions?** Check out our [Help Center](./help_center.html) or [Contact Us](./contact.html)

---

*Last Updated: February 2026*
