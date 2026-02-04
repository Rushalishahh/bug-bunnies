import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
    User, Calendar, Globe, Phone, Mail, Users,
    Stethoscope, FileText, Clock, AlertCircle,
    CheckCircle, ArrowLeft, Key
} from 'lucide-react';
import './AddPatient.css';

const AddPatient = () => {
    const navigate = useNavigate();
    const [step, setStep] = useState(1);
    const [patientId, setPatientId] = useState(null);

    const [formData, setFormData] = useState({
        // Basic Information
        fullName: '',
        dob: '',
        gender: 'Male',
        language: 'Hindi',
        contact: '',
        email: '',

        // Guardian Details
        guardianName: '',
        relationship: '',
        guardianContact: '',

        // Clinical Info
        primaryIssue: 'Articulation',
        secondaryConcerns: '',
        diagnosis: '',
        startDate: '',
        frequency: '2 sessions/week',

        // Medical & Background
        hearingIssues: 'No',
        neuroHistory: 'No',
        prevTherapy: 'No',
        notes: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (step < 4) {
            setStep(step + 1);
        } else {
            // Generate mock ID
            const newId = `SS-${Math.floor(100000 + Math.random() * 900000)}`;
            setPatientId(newId);
            setStep(5); // Success step
        }
    };

    return (
        <div className="add-patient-page">
            <div className="add-patient-container">
                <header className="add-patient-header">
                    <button onClick={() => navigate(-1)} className="back-btn">
                        <ArrowLeft size={18} /> Back
                    </button>
                    <h1>Register New Patient</h1>
                    <div className="step-indicator">
                        <div className={`step-dot ${step >= 1 ? 'active' : ''}`}>1</div>
                        <div className="step-line"></div>
                        <div className={`step-dot ${step >= 2 ? 'active' : ''}`}>2</div>
                        <div className="step-line"></div>
                        <div className={`step-dot ${step >= 3 ? 'active' : ''}`}>3</div>
                        <div className="step-line"></div>
                        <div className={`step-dot ${step >= 4 ? 'active' : ''}`}>4</div>
                    </div>
                </header>

                {step <= 4 ? (
                    <form className="patient-form" onSubmit={handleSubmit}>
                        {step === 1 && (
                            <div className="form-section">
                                <h3 className="section-title"><User size={20} /> Basic Patient Information</h3>
                                <div className="form-grid">
                                    <div className="form-group">
                                        <label>Full Name</label>
                                        <input type="text" name="fullName" placeholder="e.g. Aryan Sharma" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Date of Birth / Age</label>
                                        <input type="date" name="dob" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Gender</label>
                                        <select name="gender" onChange={handleChange}>
                                            <option>Male</option>
                                            <option>Female</option>
                                            <option>Other</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label>Primary Language</label>
                                        <input type="text" name="language" placeholder="e.g. Hindi, English, Marathi" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Contact Number</label>
                                        <input type="tel" name="contact" placeholder="+91 00000 00000" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Email ID (Optional)</label>
                                        <input type="email" name="email" placeholder="patient@example.com" onChange={handleChange} />
                                    </div>
                                </div>
                            </div>
                        )}

                        {step === 2 && (
                            <div className="form-section">
                                <h3 className="section-title"><Users size={20} /> Guardian / Caregiver Details</h3>
                                <div className="form-grid">
                                    <div className="form-group full-width">
                                        <label>Guardian Name</label>
                                        <input type="text" name="guardianName" placeholder="e.g. Sanjay Sharma" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Relationship to Patient</label>
                                        <input type="text" name="relationship" placeholder="e.g. Father" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Contact Number</label>
                                        <input type="tel" name="guardianContact" placeholder="+91 00000 00000" required onChange={handleChange} />
                                    </div>
                                </div>
                            </div>
                        )}

                        {step === 3 && (
                            <div className="form-section">
                                <h3 className="section-title"><Stethoscope size={20} /> Clinical Information</h3>
                                <div className="form-grid">
                                    <div className="form-group">
                                        <label>Primary Speech Issue</label>
                                        <select name="primaryIssue" onChange={handleChange}>
                                            <option>Articulation</option>
                                            <option>Fluency (Stuttering)</option>
                                            <option>Voice Disorder</option>
                                            <option>Aphasia</option>
                                            <option>Language Delay</option>
                                            <option>Apraxia</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label>Secondary Concerns (Optional)</label>
                                        <input type="text" name="secondaryConcerns" placeholder="e.g. Short attention span" onChange={handleChange} />
                                    </div>
                                    <div className="form-group full-width">
                                        <label>Diagnosis / Referral Notes</label>
                                        <textarea name="diagnosis" placeholder="Detailed diagnosis or notes from referring physician..." onChange={handleChange}></textarea>
                                    </div>
                                    <div className="form-group">
                                        <label>Date of Therapy Start</label>
                                        <input type="date" name="startDate" required onChange={handleChange} />
                                    </div>
                                    <div className="form-group">
                                        <label>Therapy Frequency</label>
                                        <select name="frequency" onChange={handleChange}>
                                            <option>1 session/week</option>
                                            <option>2 sessions/week</option>
                                            <option>3 sessions/week</option>
                                            <option>Daily</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                        )}

                        {step === 4 && (
                            <div className="form-section">
                                <h3 className="section-title"><FileText size={20} /> Medical & Background Notes</h3>
                                <div className="form-grid">
                                    <div className="form-group">
                                        <label>Hearing Issues?</label>
                                        <select name="hearingIssues" onChange={handleChange}>
                                            <option>No</option>
                                            <option>Yes (Mild)</option>
                                            <option>Yes (Severe)</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label>Neurological History?</label>
                                        <select name="neuroHistory" onChange={handleChange}>
                                            <option>No</option>
                                            <option>Yes</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label>Previous Speech Therapy?</label>
                                        <select name="prevTherapy" onChange={handleChange}>
                                            <option>No</option>
                                            <option>Yes (ongoing)</option>
                                            <option>Yes (completed)</option>
                                        </select>
                                    </div>
                                    <div className="form-group full-width">
                                        <label>Additional Therapist Notes (Purpose: AI Recommendations)</label>
                                        <textarea name="notes" placeholder="Enter any additional context here for better AI-assisted plan generation..." onChange={handleChange}></textarea>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="form-navigation">
                            {step > 1 && (
                                <button type="button" onClick={() => setStep(step - 1)} className="prev-btn">
                                    Previous
                                </button>
                            )}
                            <button type="submit" className="next-btn">
                                {step === 4 ? 'Register & Generate ID' : 'Continue'}
                            </button>
                        </div>
                    </form>
                ) : (
                    <div className="success-section">
                        <div className="success-icon">
                            <CheckCircle size={64} />
                        </div>
                        <h2>Patient Registered Successfully!</h2>
                        <p>Registration for <strong>{formData.fullName}</strong> is complete. Share the login ID below with the patient.</p>

                        <div className="login-id-card">
                            <div className="id-label">Patient Login ID</div>
                            <div className="id-value">{patientId}</div>
                            <button className="copy-btn">Copy ID</button>
                        </div>

                        <div className="success-actions">
                            <button onClick={() => navigate('/dashboard/therapist/patients')} className="primary-btn">
                                View Patient List
                            </button>
                            <button onClick={() => setStep(1)} className="secondary-btn">
                                Register Another Patient
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AddPatient;
