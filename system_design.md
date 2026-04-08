# 3. SYSTEM DESIGN

## 3.1 Database Design

The database for "Suhwa" uses SQLite for development and can be upgraded to PostgreSQL for production. It stores user information, activities, and model data.

### Tables:
- **users**: id (primary key), username, email, password_hash, profile_pic, created_at, last_login.
- **activities**: id, user_id (foreign key to users), activity_type (e.g., 'detection', 'upload'), timestamp, details.
- **uploads**: id, user_id, filename, filepath, upload_time, processed (boolean).
- **models**: id, name, version, filepath, created_at.

Indexes on user_id for performance.

## 3.2 Architecture Diagram

The system follows a three-tier architecture:

```
[Client Browser] <--- HTTP/HTTPS ---> [Flask Web Server]
                                      |
                                      |--- [Backend Modules: Sign Detection, Auth, Utils]
                                      |
                                      |--- [Database: SQLite/PostgreSQL]
                                      |
                                      |--- [ML Models: TensorFlow, MediaPipe]
```

- **Presentation Layer**: HTML/CSS/JS frontend.
- **Application Layer**: Flask routes, business logic.
- **Data Layer**: Database and file storage.

## 3.3 Data Flow Diagram

Level 0 DFD:

```
User Input (Video/Webcam) --> Process Video --> Detect Signs --> Translate to Text --> Output (Text/Subtitles)
                                      |
                                      |--> Store in DB --> User Profile Update
```

Detailed flow: Video captured -> MediaPipe processes landmarks -> TensorFlow model predicts -> Text generated -> Displayed to user.

## 3.4 ER Diagram

Entities and Relationships:

- **User** (1) -- (many) **Activity**
- **User** (1) -- (many) **Upload**
- **Model** (independent, referenced by system)

Attributes as in Database Design.

## 3.5 UML Diagrams

### Class Diagram:
- Classes: User, Activity, Upload, SignDetector, ModelLoader.
- Relationships: User has many Activities and Uploads; SignDetector uses ModelLoader.

### Sequence Diagram:
User logs in -> Uploads video -> System processes -> Returns translation -> Updates activity log.

### Use Case Diagram:
Actors: User, Admin.
Use cases: Register, Login, Upload Video, Live Detect, View Profile, Manage Models.
