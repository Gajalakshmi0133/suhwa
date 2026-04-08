# SUHWA: A WEB-BASED REAL-TIME SIGN LANGUAGE DETECTION AND LEARNING PLATFORM

## 1. INTRODUCTION

### 1.1 Abstract
The "Suhwa" project is a web application designed to achieve real-time detection and translation of sign language using machine learning and computer vision techniques. The system is designed to help bridge the communication gap of deaf or hard-of-hearing individuals by recognizing ASL gestures from live video streams or uploaded videos and converting them into readable text or subtitles. Built using the Flask framework, the application provides features such as live webcam detection, video upload processing, and user management, ensuring ease of use and accessibility. 

The system is integrated with MediaPipe for the accurate detection of hand landmarks and TensorFlow for the classification of gestures with deep learning. These technologies provide for fast feature extraction and high-accuracy real-time gesture recognition. Besides translation, the platform includes learning and practice modules to support sign language education. The Suhwa project was designed to be scalable, secure, and user-friendly, offering an inclusive digital solution to improve communication and accessibility for the deaf and hard-of-hearing community.

### 1.2 Problem Statement
Individuals who are deaf or hard of hearing face continuing barriers in communication, which negatively impact daily interactions, educational opportunities, and work activities due in part to a general lack of understanding about sign language. Traditional sign interpretation requires trained interpreters, which are often expensive, not readily available, and impracticable for situations requiring spontaneity or urgency. While some attempts have been made at automatically recognizing sign languages, there are considerable weaknesses with many of the currently developed approaches, including poor recognition accuracy, problems with dynamic and continuous gesture handling, no processing in real-time, and low robustness against real-life conditions. Most state-of-the-art systems continue to remain laboratory prototypes or limited mobile applications with non-user-friendly, scalable, and secure web-based interfaces. Further, most systems support a single sign language and do not provide learning, practice, and feedback from users. There is a pressing need for a real-time, accurate, and web-based sign language detection and translation system that would quickly and accurately recognize signs and express the translations in meaningful text to further improve the accessibility of communication and advance the cause of inclusivity for the DHH.

### 1.3 Objectives and Scope of the Project
The main objective of the Suhwa project is to design and develop a real-time sign language detection and translation system using machine learning techniques in order to increase the accessibility of communication for deaf or hard-of-hearing individuals. The project focuses on developing a web-based interface capable of supporting both live webcam-based gesture detection and video upload processing. It also intends to introduce secured user authentication and profile management capabilities, allowing personalization of the user experience and data security. It is designed to support multiple sign languages, starting with ASL. The proposed system contains learning, practice, and revision modules in order to develop users' skills in sign languages. It is designed with a focus on high accuracy, scalability, and user-friendliness to offer reliable performance and broad usability. 

**Objectives:** 
- A real-time sign language detection system would be developed using Machine Learning. 
- To design a web-based system for detecting live webcam and processing uploaded videos.
- Provide secure user authentication and manage user profiles. 
- Supports multiple sign languages out of the box, starting with ASL.
- Implementing secure user authentication and profile management capabilities.
- Supporting multiple sign languages, starting with American Sign Language.
- Delivering learning, practice, and revision modules to the users.
- Ensuring scalability, accuracy, and usability of the system.

**Scope of the Project:**
The project specifies the boundaries within which the system is designed and developed. Its principal aim is to recognize ASL gestures and translate them into text in real time. Gesture detection is conducted via webcams and uploaded video inputs, and the backend processes using TensorFlow and MediaPipe. The frontend of the application is built using the standard web technologies HTML, CSS, and JavaScript. Dataset preparation involves both image-based data and gesture sequence data to train the recognition models. Mobile app development and offline processing are among the features that are out of the scope of this implementation. 
- The present study focuses on the recognition of American Sign Language. 
- Gesture detection is done in real time via webcams and uploaded video content. 
- MediaPipe and TensorFlow are used for backend processing. 
- JavaScript, HTML, and CSS are used in frontend development. 
- The dataset is prepared from image data and gesture sequence data. 
- Mobile app development and offline processing are beyond the scope of this paper.

### 1.4 Project Description
The Suhwa system is a comprehensive web application which, through the integration of machine learning and computer vision methodologies, real-time recognizes and renders into text the sign language gestures performed by a user. The fundamental aim of the system is to improve the communication access for deaf and hard of hearing by converting Sign Language Gestures into legible text. The application accommodates real-time webcam feed and existing video files uploads, therefore providing adaptable deployment modalities for various environments. 

These will provide the video input and processing through the use of computer vision to detect and track hand movements. MediaPipe is used for hand landmark detection and extracting precise spatial and temporal features from gestures. These landmarks form the basic inputs to the gesture recognition pipeline, which allows for accurate representations of both static and dynamic sign language gestures. 

Following the extraction of hand landmarks, the data are processed using deep learning models running on TensorFlow, which have been previously trained on sign language datasets. These models examine the sequence of gestures to identify which patterns correspond to certain signs and make correct predictions. Subsequently, the recognized gestures are translated into meaningful text presented to the user as subtitles or textual output that enhances effective real-time communication.

The Suhwa app includes full-featured user authentication and management: user registration, login, and profile configuration are supported, allowing for secure access to the system, personalization of the user experience, and usages tracking. The platform also houses learning tutorials, practice sessions, and revision modules to facilitate the acquisition of sign language and improve gesture recognition skills through interactive feedback.

The system is designed on a modular and scalable architecture, allowing for easy maintenance and future extension. The training of new models and system configuration are possible for the integration of any other sign languages. Its web-based nature also makes it cross-device accessible without the need for specialized hardware, making Suhwa flexible and inclusive, with future compatibility in real-time sign language detection and translation.


## 2. SYSTEM STUDY

### 2.1 Existing System Study
The current state-of-the-art systems for recognizing signs lack accessibility and accuracy. Most smartphone applications accessible today recognize stationary hand gestures such as letters or signs. However, they do not recognize dynamic hand gestures essential for effective communication when carrying out signs. The effectiveness of these smartphone applications is hampered by lighting changes, the environment's background, and the speed at which the hand signal is being made.

Tools available in the current market for gesture recognition and hand tracking, such as OpenPose and MediaPipe, are very efficient in pose recognition and hand point estimation; however, they are mainly research-centric. They work very efficiently in analyzing motion patterns and extracting features; however, they do not offer a complete and ready-to-use solution in the web domain. This makes them impractical and unaffordable for common use. 

As it pertains to commercial applications, the reliance on interpretation is based on video conferencing. This may result in accurate interpretation, but it can be quite expensive and difficult to implement on a large scale. As it pertains to research and prototyping phases associated with academics, there may be a preoccupation with improving the sophistication associated with the AI analysis. The requirements based on authenticating users, learning components, secure data storage and handling, and scaling may not be appropriately considered.

### 2.2 Proposed System Study
The Suhwa system is designed to bridge the gaps in sign language technology today with a real-time, user-friendly, integrated solution. More than a suite of various applications or a set of research demos, Suhwa integrates live video processing, machine learning, and a full-stack web framework into an end-to-end platform for detecting and translating sign language. It can capture live video from a webcam or share uploaded videos; it then processes the clips through steps that identify the position of hands and sequences of gestures for exact recognition. Suhwa maintains accuracy and reliability on fluid or continuous gestures by using MediaPipe for tracking hand movements and TensorFlow for deep learning-based classification.

Beyond instant gesture recognition, the system also focuses on user-oriented features: secure sign-up, login, and profile management. The user will be able to personalize their settings, track progress, and enjoy practice and revision through interactive modules. Such elements also allow the platform to remain informative and not just functional, bringing more fluency to beginners and learners.

One thing worth noting as a unique feature of the Suhwa system is its capacity to support multiple sign languages. Even though the system begins with support for American Sign Language (ASL), its modularity means the support for the various sign languages can easily be expanded in the future.

The architecture is scalable and has been designed to support the cloud. This means it can handle multiple users concurrently and won’t result in slower performance. As a result, Suhwa can easily have a broad application, ranging from educational settings to online learning sites and community support initiatives. The combination of real-time processing capabilities, scalability, and support for multiple languages gives Suhwa a distinct advantage over the current alternatives, most of which are single-user systems, experiment-focused, or commercial.

Overall, Suhwa provides a holistic, inclusive method for sign language interpretation and translation. By combining innovative technology, interactive learning elements, and web functionality, the project not only benefits the deaf and hard of hearing communities in communicating more efficiently, but also improves education, accessibility, and inclusion on a wider societal level.

### 2.3 System Analysis
System analysis is important in the development of the Suhwa project. It explores what the system should do, if it were feasible, and exactly how the overall design would come together. By examining functional and non-functional requirements, assessing practicality, and highlighting features and key benefits of the system, this analysis helps to ensure the final system will meet user needs while remaining technically and operationally viable.

**Functional Requirements:**
The Suhwa system offers a robust set of capabilities to offer real-time sign language detection and translation.
- User Registration and Authentication: Secure login and setting up accounts for the protection of the user account.
- Real-time Webcam-based Sign Detection: Identifies the gestures from live video intake.
- Video Upload and Gesture Processing: Users can upload a recorded video for gesture recognition.
- Text Translation/Subtitle Generation: Translates detected gestures into readable text or subtitles.
- Profile Management and Activity Tracking: The students will be managing their profiles, tracking their learning or practice. 
- Learning and Practice Modules: Availability of interactive multimedia tutorials, exercises, and revision tools to develop sign language skills.

**Non-Functional Requirements:**
Beyond what the system can functionally do, it is built to meet basic standards for performance and usability:
- Low Latency: Real-time processing for smooth gesture recognition.
- High Recognition Accuracy: Reliable results from deep learning models using MediaPipe landmarks.
- Handling User Data Securely: protects sensitive personal information and activity logs.
- Responsive and intuitive user interface works across devices to create a seamless and accessible experience.
- Scalability: Handles many concurrent users with no slowdowns.

**Feasibility Analysis:**
Considering the following reasons, the proposed system is viable and implementable:
- Technical Feasibility: Can be realized using open-source tools such as Flask for the backend, TensorFlow for machine learning, and MediaPipe for hand tracking.
- Economic Feasibility: Low cost in development because it has free, open-source frameworks and libraries.
- Operational Feasibility: Easy to deploy and maintain; web-based design runs across multiple platforms without special hardware.

**Key Features:**
Suwha provides a range of advanced functionalities to enhance its ease of use and performance:
- Real-time Gesture Recognition: It can operate in real time for both static and dynamic sign-language gestures.
- Precise Translation to Text/ Subtitles: Translates the gestures into readable output for comprehensible communication.
- User Authentication and Profile Management: Secure, personalized user experience. 
- Interactive Learning Modules: Tutorials, practice exercises, and revision tools to build skills. 
- Modular and Scalable Architecture: Allows for the easy addition of more sign languages as well as future expansion.

**Advantages:**
- Reduction in dependence on human interpreters.
- Improving communication for Deaf and hard-of-hearing individuals.
- An accessible and user-friendly digital platform.
- Facilitates Learning & Skill Acquisition.
- Scalability, real-time, accurate processing.


### 2.4 Hardware Requirements
Suhwa’s real-time sign language recognition and translation is a direct video feed processing task, and a sound computational performance setup is a requirement for smooth and precise functioning. A multicore experience with Intel i7 or similar specifications is what is required for simultaneous management of video and model processing. Some power-packed memory is also a must. A memory setup with 16GB RAM is therefore a worthwhile requirement since it will not let data processing and calculations slow down.
For creating as well as executing the Suhwa project, the hardware environment is designed to facilitate real-time sign language recognition as well as translation.

- **Server**: The processing would take place in a multi-core computer (consider an Intel i7 or similar). It would feature 16GB RAM so that video processing, gestures, as well as machine learning inference, can occur smoothly. There would also be a separate GPU, including at least an NVIDIA GTX 1060, which would significantly reduce latency in real-time gesture recognition.
- **Client Devices**: These devices include webcams attached to computers or tablets as long as the computers possess an internet connection for the online detection component.
- **Storage**: Storage requirements should be considered for a least 50GB of storage for data sets, trained models, and logs. Storage capacity will be required for gesture data and model storage in substantial size, and thus storage can be updated when needed.
- **Development Environment**: This development is done on a normal computer with Python in Windows 11. However, it can run on different operating systems as it is a cross-platform.


### 2.5 Software Requirements
The Suhwa System is a web application that uses machine learning to run with a carefully honed technology stack in order to smoothly develop, deploy, and maintain the System. The technology requirements run the gamut from operating systems and languages to frameworks and libraries, and finally to the backend and front-end technologies.

- **Operating System**: The development environment supports both Windows 10/11, Linux (Ubuntu), and macOS to ensure a cross-platform interface. Though we developed on the Windows 11 platform, Suhwa works perfectly on a Linux and macOS server too. Therefore, our system works seamlessly on different environments. 
- **Programming Languages**: Suwha’s underlying logic and machine learning models are coded with either Python 3.8 or a later version and also use efficient tools for data processing, computer vision, and deep learning. JavaScript is used for creating dynamic and interactive user interface pages for webcam detection, uploading videos, and creating learning modules.
- **Frameworks**: On the backend side, we rely on Flask, a lightweight framework that deals effectively with routing, API, as well as server logic. For the models used in gesture recognition, TensorFlow 2.x serves as the backbone, together with MediaPipe, which detects landmarks on a hand to recognize gestures effectively.
- **Libraries**: From the library side, we use OpenCV for image and video processing, NumPy for numerical computation, and Pillow for image processing. These libraries work in tandem to process the videos, extract the hand landmarks, and prepare the inputs for the models.
- **Storage**: SQLite is used as a small relational database for storing data in development. In production, PostgreSQL is the recommended database to handle multiple users, certain data, and concurrent operations effectively.
- **Web Technologies**: The frontend is developed in HTML5, CSS3, and JavaScript technology, providing a responsive, interactive, and user-friendly interface. Using HTML5, CSS3, and JavaScript technology makes it compatible with all devices, such as PCs, Laptops, and Tablets, with the aid of browsers.


### 2.6 Software Tools Description
The Suhwa project is based on an ecosystem of software tools and frameworks that collaborate with each other to offer real-time sign language recognition, translation, and interaction. All the software tools play their distinctive role in ensuring the system is fast, scalable, and user-friendly.

- **Flask**: A lightweight Python-based web framework, responsible for the backend part of Suhwa, routing, API, and server-level functionality, connecting the frontend GUI to the ML models. It has been found to be ideal for quick web app developments and also has the potential for scaling in the future.
- **TensorFlow**: Open-source machine learning platform that can develop, train, and deploy deep learning models. In the SuHwa system, TensorFlow supports models analyzing hand landmark information and making accurate gesture classifications. Using the GPU makes TensorFlow capable of processing predictions in real-time. 
- **MediaPipe**: A cross-platform framework for designing machine learning pipelines, especially in computer vision. In this case, it deals with the tracking of hands and landmarks. In the end, it provides the precise coordinates of the joints of the hands, which are used in models for gesture recognition.
- **OpenCV**: OpenCV remains one of the most important computer vision toolkits for anything that has to do with video or images. In Suhwa, it takes care of grabbing frames from webcams, cleaning and preparing images, and weaving video input into the gesture-recognition workflow.
- **NumPy**: NumPy is a Python library for numbers, powering the processing of hand landmark arrays, executing mathematical operations, and preparing data for machine-learning models. Its fast, optimized computations are essential for smooth real-time performance.
- **Pillow**: Pillow is a python library and is considered the successor of the Python Imaging Library. Preprocessing for images includes rescaling, resizing, cropping, and formatting, and it ensures that video frames and datasets are in the right shapes and formats for training and inference. 
- **SQLite / PostgreSQL**: SQLite is a light relational database used in the developmental stage to store user data, activity logs, and session records. In a production environment, it is highly recommended to use PostgreSQL because it guarantees better durability, scalability, and handling of concurrent users. 
- **Git**: Git is used for version control; hence it's the back that controls the codebase, tracks the changes, and helps in effective collaboration. It maintains the integrity of code, enables the movement to previous states, and facilitates teamwork on the software project.
- **VS Code**: Visual Studio Code serves as your code-editing and development environment in which you develop and debug your project. The presence of in-built support for both Python and Git in Visual Studio Code would make it a strong base for web development and machine learning.
- **Jupyter Notebook**: Jupyter Notebook is where all the experiments occur, where data mining takes place, as well as all analyses. This is where you can experiment with deep learning models, data for hand landmarks, as well as model parameters prior to pushing everything to the web app. 


## 3. DESIGN OF THE SYSTEM
The Suhwa system is a robust, web-based solution for real-time sign language detection and translation. Having modularity, scalability, and performance in mind, it delivers smooth client-side operation, efficient server processing, and accurate machine learning inferences. This section takes a deeper dive into the database layout, total architecture, data flow, entity-relationship framework, and UML diagrams to give a good overview of the system's inner structure and functioning.

### 3.1 Database Design
The database forms the core of Suhwa: storing user accounts, activity logs, video uploads, and machine learning models. Development uses SQLite for simplicity and speed in setup, while it recommends PostgreSQL for production to allow for large numbers of concurrent users and to support stronger scalability. Proper indexing, foreign key constraints, and normalization are used to ensure data integrity, performance, and maintainability in the schema.

**Tables and Structure:**
- **users**: Stores information about registered users.
    - Fields: id (Primary Key), name, username, email, password_hash, created_at.
- **[user]**: Stores detailed profile information for users (fallback/extended table).
    - Fields: id, username, password, email, first_name, last_name, dob, gender, is_confirmed, avatar, preferred_language, total_signs, created_at, last_login.
- **detection_history**: Logs all user sign detection actions.
    - Fields: id, user_id (Foreign Key), timestamp, text, confidence, raw_label.

The indexes on user_id in detection_history can speed up the relevant queries. This design allows for efficient storage and retrieval and updating of system and user data, including both the operational and learning components of Suhwa.

### 3.2 Architecture Design
Suhwa has a three-tier structure, which separates the structure into presentation, application, and data. This increases modularity, maintenance, and scalability, as well as efficient real-time processing and database management.

**Text Diagram:**
```
[Client Browser] <--- HTTP/HTTPS/WebSocket ---> [Flask Web Server]
                                                  |
                                                  |--- [Backend Modules: Sign Detection, Authentication, Utilities]
                                                  |
                                                  |--- [Database: SQLite/PostgreSQL]
                                                  |
                                                  |--- [ML Models: TensorFlow, MediaPipe]
```

**1. Presentation Layer (Frontend)**
Suhwa's frontend is the part of Suhwa users see: implemented with HTML5, CSS3, and JavaScript. It should be responsive, smooth, and easy to work with. User can do the following:
- See live webcam detection: for real-time gesture recognition
- Upload pre-recorded videos for analysis
- See translations as text or captions
- Share and use learning and practice modules
- Manage profiles and activity history

**2. Application Layer (Backend)**
The backend layer represents the business logic, routing, and coordination between the frontend, database, and machine learning components. Built using the Flask framework, it provides:
- **Routing**: It is used to forward the user request such as login, video upload, live detection to the appropriate backend functions.
- **Authentication**: This protects accounts with hashed passwords, session management.
- **Gesture Processing Logic**: This uses MediaPipe for tracking the hand, extracting landmarks, and TensorFlow for gesture classification. 
- **Utilities and Services**: This layer encapsulates file handling, video preprocessing, activity logging, and error handling.

**3. Data Layer (Database & Storage)**
The handling of all persistent data, from storing it through retrieval and organization, is the function of the Data Layer.
- **User Information**: The details provided for account registration, user information, and authentication details.
- **Activity Log**: detection history, progress, and uploaded videos.
- **Uploaded Videos**: Original video files uploaded for processing.
- **Machine Learning Models**: TensorFlow-trained models and MediaPipe settings data.

**4. Machine Learning Integration**
- MediaPipe uses real-time video frame analysis to detect hand landmarks.
- TensorFlow models use these landmarks for classification into corresponding sign language signs. 
- Translates gestures into text or subtitles displayed on the frontend.

**5. Communication Across Layers**
Layer-to-layer conversation happens through HTTP/HTTPS and WebSockets:
- Frontend sends video streams or uploaded files to the Flask Backend.
- Backend processes data, computes ML inferences, and stores results in the database.
- Output (translation) is returned to the frontend via WebSockets (for real-time) or HTTP.
