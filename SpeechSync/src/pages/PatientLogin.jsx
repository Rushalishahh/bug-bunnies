import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Key, User, Lock, ArrowRight, ArrowLeft, CheckCircle } from 'lucide-react';
import './PatientLogin.css';

const PatientLogin = () => {
    const navigate = useNavigate();
    const [step, setStep] = useState(1); // 1: ID, 2: Password Setup
    const [loginId, setLoginId] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');

    const handleNext = (e) => {
        e.preventDefault();
        if (loginId.startsWith('SS-')) {
            setStep(2);
        } else {
            alert('Please enter a valid Patient ID (e.g., SS-123456)');
        }
    };

    const handleFinalSubmit = (e) => {
        e.preventDefault();
        if (password === confirmPassword && password.length >= 6) {
            // Mock registration - In real app, save to backend
            navigate('/dashboard/patient', { state: { loginId, name: 'Arjun Verma' } });
        } else {
            alert('Passwords do not match or are too short (min 6 chars)');
        }
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <header className="login-header">
                    <Link to="/" className="login-logo">S</Link>
                    <h1>Patient Portal</h1>
                    <p>{step === 1 ? 'Activate your account with your Patient ID' : 'Create a secure password to continue'}</p>
                </header>

                <div className="login-card">
                    {step === 1 ? (
                        <form onSubmit={handleNext} className="login-form">
                            <div className="form-group">
                                <label>Enter Patient Login ID</label>
                                <div className="input-with-icon">
                                    <Key size={18} />
                                    <input
                                        type="text"
                                        placeholder="SS-XXXXXX"
                                        value={loginId}
                                        onChange={(e) => setLoginId(e.target.value.toUpperCase())}
                                        required
                                    />
                                </div>
                                <span className="input-hint">This ID was provided by your therapist.</span>
                            </div>
                            <button type="submit" className="login-btn">
                                Verify ID <ArrowRight size={18} />
                            </button>
                        </form>
                    ) : (
                        <form onSubmit={handleFinalSubmit} className="login-form">
                            <div className="form-group">
                                <label>Create Password</label>
                                <div className="input-with-icon">
                                    <Lock size={18} />
                                    <input
                                        type="password"
                                        placeholder="Min 6 characters"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                    />
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Confirm Password</label>
                                <div className="input-with-icon">
                                    <Lock size={18} />
                                    <input
                                        type="password"
                                        placeholder="Repeat password"
                                        value={confirmPassword}
                                        onChange={(e) => setConfirmPassword(e.target.value)}
                                        required
                                    />
                                </div>
                            </div>
                            <button type="submit" className="login-btn">
                                Create Account <CheckCircle size={18} />
                            </button>
                            <button type="button" onClick={() => setStep(1)} className="back-link">
                                <ArrowLeft size={16} /> Back to ID entry
                            </button>
                        </form>
                    )}
                </div>

                <div className="login-footer">
                    <p>Having trouble? Contact your therapist or <a href="#">Support</a></p>
                </div>
            </div>
        </div>
    );
};

export default PatientLogin;
