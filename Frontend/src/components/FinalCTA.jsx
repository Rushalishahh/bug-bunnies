import React from 'react';
import { Link } from 'react-router-dom';
import './FinalCTA.css';

const FinalCTA = () => {
    return (
        <section id="get-started" className="final-cta section-padding">
            <div className="container">
                <div className="cta-card">
                    <h2 className="cta-title">Start your speech therapy journey with clarity</h2>
                    <p className="cta-description">Join thousands of patients and therapists already using SpeechSync to achieve their goals.</p>
                    <Link to="/get-started" className="btn btn-primary btn-large">Get Started Today — It's Free</Link>
                    <p className="cta-login-prompt">Already have an account? <Link to="/portal">Login here</Link></p>
                </div>
            </div>
        </section>
    );
};

export default FinalCTA;
