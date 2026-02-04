import React from 'react';
import { Link } from 'react-router-dom';
import './PortalSelection.css';

const PortalSelection = () => {
    return (
        <div className="portal-selection-container">
            <div className="portal-header">
                <div className="logo-section">
                    <div className="main-logo-icon">S</div>
                    <h1>SpeechSync</h1>
                    <p className="subtitle">AI-Powered Progress Tracking & Interactive Practice</p>
                </div>
            </div>

            <div className="portals-grid">
                {/* Therapist Portal */}
                <div className="portal-card">
                    <div className="portal-icon therapist-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                            <circle cx="9" cy="7" r="4" />
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                            <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                        </svg>
                    </div>
                    <h2>Therapist Portal</h2>
                    <p className="card-desc">Manage patients, track progress, and assign exercises</p>
                    <ul className="feature-list">
                        <li>Create and manage patient profiles</li>
                        <li>Record baseline and track sessions</li>
                        <li>Assign AI-suggested exercises</li>
                        <li>View detailed progress dashboards</li>
                    </ul>
                    <div className="portal-actions">
                        <Link to="/signup/therapist" className="portal-btn therapist-btn">
                            Sign Up <span>→</span>
                        </Link>
                        <Link to="/login/therapist" className="portal-btn therapist-btn-outline">
                            Login <span>→</span>
                        </Link>
                    </div>
                </div>

                {/* Patient Portal */}
                <div className="portal-card">
                    <div className="portal-icon patient-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                            <circle cx="12" cy="7" r="4" />
                        </svg>
                    </div>
                    <h2>Patient Portal</h2>
                    <p className="card-desc">Practice exercises and track your progress</p>
                    <ul className="feature-list">
                        <li>Complete assigned exercises</li>
                        <li>Interactive speaking practice</li>
                        <li>Record and submit practice sessions</li>
                        <li>View your progress and achievements</li>
                    </ul>
                    <Link to="/login/patient" className="portal-btn patient-btn">
                        Login as a Patient <span>→</span>
                    </Link>
                </div>
            </div>

            <div className="back-link">
                <Link to="/">← Back to Home</Link>
            </div>
        </div>
    );
};

export default PortalSelection;
