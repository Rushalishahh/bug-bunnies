// app.js
const express = require("express");
const mongoose = require("mongoose");
const path = require("path");
const fs = require("fs");
const multer = require("multer");

// Import models
const Therapist = require("./models/Therapist");
const Patient = require("./models/Patient");
const Session = require("./models/Session");
const Audio = require("./models/Audio");

const app = express();

/* --------------------------------------------------
   ENSURE UPLOAD FOLDER EXISTS
-------------------------------------------------- */
const uploadDir = path.join(__dirname, "uploads/audios");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

/* --------------------------------------------------
   MULTER CONFIG
-------------------------------------------------- */
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const safeName = file.originalname.replace(/\s+/g, "_");
    cb(null, Date.now() + "-" + safeName);
  }
});

const upload = multer({
  storage,
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();

    if (
      file.mimetype.startsWith("audio/") ||
      [".wav", ".mp3", ".aac", ".ogg"].includes(ext)
    ) {
      cb(null, true);
    } else {
      cb(new Error("Only audio files allowed"));
    }
  }
});

/* --------------------------------------------------
   MIDDLEWARE
-------------------------------------------------- */
app.use(express.json());
app.use("/uploads", express.static("uploads"));

/* --------------------------------------------------
   DB CONNECTION
-------------------------------------------------- */
mongoose
  .connect("mongodb://127.0.0.1:27017/therapyDB")
  .then(() => console.log("MongoDB connected"))
  .catch(err => console.error(err));

/* --------------------------------------------------
   ROUTES
-------------------------------------------------- */

// Add therapist
app.post("/therapists", async (req, res) => {
  try {
    const therapist = new Therapist(req.body);
    await therapist.save();
    res.json(therapist);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Add patient
app.post("/patients", async (req, res) => {
  try {
    const patient = new Patient(req.body);
    await patient.save();
    res.json(patient);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Add session manually
app.post("/sessions", async (req, res) => {
  try {
    const session = new Session(req.body);
    await session.save();
    res.json(session);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Upload audio + create session
app.post("/upload-audio", upload.single("audio"), async (req, res) => {
  try {
    const {
      therapistId,
      patientId,
      speaker,
      index,
      score,
      baseline,
      notes
    } = req.body;

    if (!req.file) {
      return res.status(400).json({ error: "Audio file required" });
    }

    // Save audio metadata
    const audio = new Audio({
      therapistId,
      audioUrl: req.file.path,
      originalName: req.file.originalname
    });
    await audio.save();

    // Create session
    const session = new Session({
      therapistId,
      patientId,
      speaker,
      sessionFile: req.file.filename,
      index,
      score,
      baseline: baseline === "true",
      notes
    });
    await session.save();

    res.status(201).json({
      message: "Audio uploaded and session saved",
      audio,
      session
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

/* --------------------------------------------------
   GLOBAL ERROR HANDLER (IMPORTANT)
-------------------------------------------------- */
app.use((err, req, res, next) => {
  res.status(400).json({ error: err.message });
});

/* --------------------------------------------------
   START SERVER
-------------------------------------------------- */
app.listen(3000, () => {
  console.log("Server running on port 3000");
});
