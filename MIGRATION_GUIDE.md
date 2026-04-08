# Community Feature - Migration Guide

This guide will help you set up the new **Live Chat & Community** feature with sample data.

## Step 1: Run Database Migration

Execute the SQL schema update to create the new tables:

```sql
-- Run in SQL Server Management Studio or similar
-- This adds the room_members table for tracking which users are in which rooms
-- File: schema.sql (lines 113-122)
```

Or directly with the schema.sql file:
```bash
sqlcmd -S your_server -d your_database -i schema.sql
```

## Step 2: Run Sample Data Migration

This will add **10 sample users** and **5 default chat rooms**.

### Option A: Using Python (Recommended)

```bash
cd e:\gajalakshmi\project\Suhwa
python migrate_sample_data.py
```

**Output:**
```
🚀 Starting migration...

✅ 10 sample users added successfully!
✅ 5 default rooms created successfully!

✨ Migration complete!
```

### Option B: Manual SQL

If you prefer, run these SQL commands:

```sql
-- Add sample users
INSERT INTO [user] (username, email, first_name, last_name, password, is_confirmed, created_at)
VALUES 
('alice_sign', 'alice@suhwa.com', 'Alice', 'Johnson', 'hashed_password', 1, CURRENT_TIMESTAMP),
('bob_practice', 'bob@suhwa.com', 'Bob', 'Smith', 'hashed_password', 1, CURRENT_TIMESTAMP),
('carol_learner', 'carol@suhwa.com', 'Carol', 'Williams', 'hashed_password', 1, CURRENT_TIMESTAMP),
('david_tutor', 'david@suhwa.com', 'David', 'Brown', 'hashed_password', 1, CURRENT_TIMESTAMP),
('emma_helper', 'emma@suhwa.com', 'Emma', 'Davis', 'hashed_password', 1, CURRENT_TIMESTAMP),
('frank_mentor', 'frank@suhwa.com', 'Frank', 'Miller', 'hashed_password', 1, CURRENT_TIMESTAMP),
('grace_expert', 'grace@suhwa.com', 'Grace', 'Wilson', 'hashed_password', 1, CURRENT_TIMESTAMP),
('henry_student', 'henry@suhwa.com', 'Henry', 'Moore', 'hashed_password', 1, CURRENT_TIMESTAMP),
('iris_teacher', 'iris@suhwa.com', 'Iris', 'Taylor', 'hashed_password', 1, CURRENT_TIMESTAMP),
('jack_enthusiast', 'jack@suhwa.com', 'Jack', 'Anderson', 'hashed_password', 1, CURRENT_TIMESTAMP);

-- Add default rooms
INSERT INTO chat_rooms (room_name, description, room_type, created_at)
VALUES 
('General Discussion', 'Welcome to the general discussion room. Practice sign conversations and ask questions!', 'public', CURRENT_TIMESTAMP),
('Alphabet Practice', 'Practice and master the sign language alphabet with fellow learners.', 'topic', CURRENT_TIMESTAMP),
('Numbers & Counting', 'Learn and practice sign language numbers and counting techniques.', 'topic', CURRENT_TIMESTAMP),
('Daily Words', 'Discover and discuss common everyday words and phrases in sign language.', 'topic', CURRENT_TIMESTAMP),
('Doubt Discussion', 'Ask questions, clear doubts, and get help from experienced signers.', 'topic', CURRENT_TIMESTAMP);
```

## Step 3: Verify Installation

Check that everything is set up correctly:

```sql
-- Check users
SELECT COUNT(*) FROM [user] WHERE username LIKE '%_sign' OR username LIKE '%_practice' OR username LIKE '%_learner';
-- Should return: 10

-- Check rooms
SELECT COUNT(*) FROM chat_rooms WHERE room_type IN ('public', 'topic');
-- Should return: 5
```

## Step 4: Test the Feature

1. **Login** to your Suhwa application
2. **Navigate** to "Community" in the main menu
3. **Verify** you can see:
   - 5 default rooms in the left sidebar
   - List of online users
   - Ability to send messages

4. **Create a Private Room**:
   - Click "Create Room" button
   - Choose "Private (Select members)"
   - Select users from the list
   - Create the room

## Sample User Credentials

All sample users have the password: `Password123`

| Username | Email | Name |
|----------|-------|------|
| alice_sign | alice@suhwa.com | Alice Johnson |
| bob_practice | bob@suhwa.com | Bob Smith |
| carol_learner | carol@suhwa.com | Carol Williams |
| david_tutor | david@suhwa.com | David Brown |
| emma_helper | emma@suhwa.com | Emma Davis |
| frank_mentor | frank@suhwa.com | Frank Miller |
| grace_expert | grace@suhwa.com | Grace Wilson |
| henry_student | henry@suhwa.com | Henry Moore |
| iris_teacher | iris@suhwa.com | Iris Taylor |
| jack_enthusiast | jack@suhwa.com | Jack Anderson |

## Default Rooms

| Room Name | Type | Description |
|-----------|------|-------------|
| General Discussion | Public | Welcome to the general discussion room |
| Alphabet Practice | Topic | Practice the sign language alphabet |
| Numbers & Counting | Topic | Learn sign language numbers |
| Daily Words | Topic | Discover everyday phrases |
| Doubt Discussion | Topic | Ask questions and get help |

## Features Enabled

✅ **Public Chat Rooms** - Community-wide discussions
✅ **Private Rooms** - Group chats with selected members
✅ **Private Messages** - One-on-one direct messaging
✅ **Sign-to-Text Chat** - Capture signs and auto-insert into messages
✅ **Online Users** - See who's online in real-time
✅ **Room Creation** - Users can create their own rooms and invite others

## Troubleshooting

**Issue:** "No users found" in member selection
- **Solution:** Run `migrate_sample_data.py` again

**Issue:** "Room creation fails"
- **Solution:** Check that all required tables exist: `chat_rooms`, `room_members`, `messages`, `private_messages`, `online_users`

**Issue:** Sign detection not working
- **Solution:** Ensure `/api/predict-image` endpoint is working properly

## Next Steps

- Add more sample data as needed
- Customize room descriptions and types
- Set up notification system for new messages
- Add room moderation features
