import React from 'react';
import './Footer.css';

const Footer = () => {
    return (
        <footer className="footer section-padding">
            <div className="container">
                <div className="footer-grid">
                    <div className="footer-brand">
                        <h3 className="footer-logo">SpeechSync</h3>
                        <p className="footer-tagline">Advanced AI-assisted speech therapy for clinics, therapists, and patients.</p>
                    </div>
                    <div className="footer-links">
                        <div className="link-group">
                            <h4>Platform</h4>
                            <a href="#how-it-works">How It Works</a>
                            <a href="#features">Features</a>
                            <a href="#pricing">Pricing</a>
                        </div>
                        <div className="link-group">
                            <h4>Company</h4>
                            <a href="#about">About</a>
                            <a href="#contact">Contact</a>
                            <a href="#careers">Careers</a>
                        </div>
                        <div className="link-group">
                            <h4>Legal</h4>
                            <a href="#privacy">Privacy Policy</a>
                            <a href="#terms">Terms of Service</a>
                        </div>
                    </div>
                </div>
                <div className="footer-bottom">
                    <p>&copy; {new Date().getFullYear()} SpeechSync. All rights reserved.</p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
