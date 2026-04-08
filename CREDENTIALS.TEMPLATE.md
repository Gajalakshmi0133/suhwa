# Credentials Template

This is a TEMPLATE file. For actual credentials, see `CREDENTIALS.md` (which is in .gitignore).

## Admin Account

```
Username: admin_user
Email: admin@suhwa.com
Password: [See CREDENTIALS.md]
```

## Sample Users (10)

All sample users use the same password for testing.

```
Password: [See CREDENTIALS.md]

Users:
- alice_sign (alice@suhwa.com)
- bob_practice (bob@suhwa.com)
- carol_learner (carol@suhwa.com)
- david_tutor (david@suhwa.com)
- emma_helper (emma@suhwa.com)
- frank_mentor (frank@suhwa.com)
- grace_expert (grace@suhwa.com)
- henry_student (henry@suhwa.com)
- iris_teacher (iris@suhwa.com)
- jack_enthusiast (jack@suhwa.com)
```

## How to Get Actual Credentials

1. Check the `CREDENTIALS.md` file in the project root
2. This file is NOT tracked by Git (see .gitignore)
3. If you don't have it, run: `python migrate_sample_data.py`

## Important Security Notes

- **NEVER** commit `CREDENTIALS.md` to version control
- **NEVER** share passwords in Slack, email, or chat
- **ALWAYS** use environment variables for production credentials
- **ALWAYS** use strong, unique passwords in production
- **ALWAYS** rotate admin password after initial setup

## Setting Up Your Own Credentials

1. Copy this template: `cp CREDENTIALS.TEMPLATE.md CREDENTIALS.md`
2. Edit `CREDENTIALS.md` with your actual credentials
3. Keep it local and NEVER commit it
4. Add to `.gitignore` (already done)

---

**Safe to commit**: ✅ This template file  
**NOT safe to commit**: ❌ CREDENTIALS.md (contains actual passwords)
