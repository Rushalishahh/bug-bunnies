import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { User, Mail, Phone, Lock, Award, Briefcase, Building, ShieldCheck, CheckCircle } from 'lucide-react';
import './TherapistSignup.css';

const TherapistSignup = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        fullName: '',
        email: '',
        mobile: '',
        password: '',
        role: 'Speech Therapist',
        qualification: '',
        experience: '',
        licenseNumber: '',
        clinicName: '',
        termsAccepted: false,
        consentData: false,
        consentDecisions: false
    });

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        navigate('/dashboard/therapist', {
            state: {
                name: formData.fullName,
                clinic: formData.clinicName,
                isDemo: false
            }
        });
    };

    return (
        <div className="signup-page">
            <div className="signup-container">
                <div className="signup-header">
                    <Link to="/" className="signup-logo">S</Link>
                    <h1>Join SpeechSync</h1>
                    <p>Create your professional therapist account</p>
                </div>

                <form onSubmit={handleSubmit} className="signup-form">
                    <div className="form-section">
                        <h3 className="section-title"><User size={20} /> Basic Personal Information</h3>
                        <div className="form-grid">
                            <div className="form-group">
                                <label>Full Name</label>
                                <div className="input-with-icon">
                                    <User size={18} />
                                    <input
                                        type="text"
                                        name="fullName"
                                        placeholder="Dr. Priya Nair"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Email Address</label>
                                <div className="input-with-icon">
                                    <Mail size={18} />
                                    <input
                                        type="email"
                                        name="email"
                                        placeholder="priya@clinic.in"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Mobile Number</label>
                                <div className="input-with-icon">
                                    <Phone size={18} />
                                    <input
                                        type="tel"
                                        name="mobile"
                                        placeholder="+91 98765 43210"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Password</label>
                                <div className="input-with-icon">
                                    <Lock size={18} />
                                    <input
                                        type="password"
                                        name="password"
                                        placeholder="••••••••"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="form-section">
                        <h3 className="section-title"><Briefcase size={20} /> Professional Details</h3>
                        <div className="form-grid">
                            <div className="form-group">
                                <label>Professional Role</label>
                                <div className="input-with-icon">
                                    <Award size={18} />
                                    <select name="role" onChange={handleChange} value={formData.role}>
                                        <option>Speech Therapist</option>
                                        <option>Speech-Language Pathologist</option>
                                        <option>Audiologist</option>
                                    </select>
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Highest Qualification</label>
                                <div className="input-with-icon">
                                    <Award size={18} />
                                    <input
                                        type="text"
                                        name="qualification"
                                        placeholder="e.g. BASLP, M.Sc Speech Therapy"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Years of Experience</label>
                                <div className="input-with-icon">
                                    <Briefcase size={18} />
                                    <input
                                        type="number"
                                        name="experience"
                                        placeholder="5"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label>License / Registration Number (Optional)</label>
                                <div className="input-with-icon">
                                    <ShieldCheck size={18} />
                                    <input
                                        type="text"
                                        name="licenseNumber"
                                        placeholder="Reg #12345"
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                            <div className="form-group full-width">
                                <label>Clinic / Hospital Name</label>
                                <div className="input-with-icon">
                                    <Building size={18} />
                                    <input
                                        type="text"
                                        name="clinicName"
                                        placeholder="Arogya Speech Center"
                                        required
                                        onChange={handleChange}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="form-section">
                        <h3 className="section-title"><ShieldCheck size={20} /> Verification & Consent</h3>
                        <div className="checkbox-group">
                            <label className="checkbox-item">
                                <input
                                    type="checkbox"
                                    name="termsAccepted"
                                    required
                                    onChange={handleChange}
                                />
                                <span className="checkmark"></span>
                                <span className="label-text">Agreement to Terms & Conditions</span>
                            </label>
                            <label className="checkbox-item">
                                <input
                                    type="checkbox"
                                    name="consentData"
                                    required
                                    onChange={handleChange}
                                />
                                <span className="checkmark"></span>
                                <span className="label-text">Consent for secure storage of patient data</span>
                            </label>
                            <label className="checkbox-item">
                                <input
                                    type="checkbox"
                                    name="consentDecisions"
                                    required
                                    onChange={handleChange}
                                />
                                <span className="checkmark"></span>
                                <span className="label-text">Confirmation of therapist-led final decisions (AI is assistive, not autonomous)</span>
                            </label>
                        </div>
                    </div>

                    <button type="submit" className="signup-submit-btn">
                        Complete Registration <CheckCircle size={20} />
                    </button>
                </form>

                <p className="login-prompt">
                    Already have an account? <Link to="/login">Login here</Link>
                </p>
            </div>
        </div>
    );
};

export default TherapistSignup;
