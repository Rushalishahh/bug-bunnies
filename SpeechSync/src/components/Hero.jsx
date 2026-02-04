import React from 'react';
import { Link } from 'react-router-dom';
import './Hero.css';

const Hero = () => {
    return (
        <section className="hero">
            <div className="container hero-grid">
                <div className="hero-content">
                    <h1 className="hero-title">SpeechSync</h1>
                    <h2 className="hero-subtitle">Because every voice deserves clarity</h2>
                    <p className="hero-description">
                        Experience the future of speech therapy with our AI-powered platform.
                        Tailored exercises, real-time feedback, and comprehensive progress tracking
                        designed for clinics, therapists, and patients.
                    </p>
                    <div className="hero-actions">
                        <Link to="/get-started" className="btn btn-primary">Get Started</Link>
                        <a href="#demo" className="btn btn-secondary">Request a Demo</a>
                    </div>
                </div>
                <div className="hero-image-container">
                    <div className="hero-image-placeholder">
                        <div className="pulse-circle"></div>
                        <div className="waveform">
                            <span></span><span></span><span></span><span></span><span></span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Hero;
