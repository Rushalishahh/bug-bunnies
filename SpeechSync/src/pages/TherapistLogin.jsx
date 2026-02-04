import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Lock, ArrowRight, ArrowLeft, CheckCircle } from 'lucide-react';
import './TherapistLogin.css';

const TherapistLogin = () => {
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        loginId: '',
        password: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        // Redirecting to dashboard with Demo data as requested
        navigate('/dashboard/therapist', {
            state: {
                name: 'Dr. Rajesh Gupta',
                clinic: 'Arogya Speech Center',
                isDemo: true
            }
        });
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <Link to="/portal" className="back-to-portal">
                    <ArrowLeft size={18} /> Back
                </Link>

                <div className="login-header">
                    <div className="login-logo-icon">S</div>
                    <h1>Therapist Login</h1>
                    <p>Access your professional dashboard</p>
                </div>

                <form onSubmit={handleSubmit} className="login-form">
                    <div className="form-group">
                        <label>Login ID / Email</label>
                        <div className="input-field">
                            <Mail size={18} />
                            <input
                                type="text"
                                name="loginId"
                                placeholder="therapist@speechsync.com"
                                required
                                value={formData.loginId}
                                onChange={handleChange}
                            />
                        </div>
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <div className="input-field">
                            <Lock size={18} />
                            <input
                                type="password"
                                name="password"
                                placeholder="••••••••"
                                required
                                value={formData.password}
                                onChange={handleChange}
                            />
                        </div>
                    </div>

                    <div className="login-options">
                        <label className="remember-me">
                            <input type="checkbox" /> Remember me
                        </label>
                        <a href="#" className="forgot-password">Forgot password?</a>
                    </div>

                    <button type="submit" className="login-submit-btn">
                        Login to Dashboard <ArrowRight size={18} />
                    </button>
                </form>

                <p className="signup-redirect">
                    New to SpeechSync? <Link to="/signup/therapist">Create an account</Link>
                </p>
            </div>
        </div>
    );
};

export default TherapistLogin;
