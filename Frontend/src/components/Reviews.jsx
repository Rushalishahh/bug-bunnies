import React from 'react';
import './Reviews.css';

const reviewsData = [
    {
        name: "Anjali Sharma",
        role: "Parent",
        text: "SpeechSync has been a game-changer for my son. The AI feedback is so encouraging, and we can actually see his progress in the weekly charts.",
        avatar: "A"
    },
    {
        name: "Rohan Mehra",
        role: "Patient",
        text: "As an adult in speech therapy, I appreciate the professional tone and the data-driven approach. It helps me stay motivated between clinic visits.",
        avatar: "R"
    },
    {
        name: "Dr. Kavita Iyer",
        role: "Speech Therapist",
        text: "The integration of AI diagnostics with traditional therapy methods makes the sessions much more efficient. My patients love using it at home.",
        avatar: "K"
    }
];

const Reviews = () => {
    return (
        <section id="reviews" className="reviews section-padding">
            <div className="container">
                <div className="section-header">
                    <h2 className="section-title">What Our Community Says</h2>
                    <p className="section-subtitle">Real progress from real people</p>
                </div>
                <div className="reviews-grid">
                    {reviewsData.map((review, index) => (
                        <div key={index} className="review-card">
                            <div className="quote-icon">“</div>
                            <p className="review-text">{review.text}</p>
                            <div className="review-author">
                                <div className="author-avatar">{review.avatar}</div>
                                <div className="author-info">
                                    <h4 className="author-name">{review.name}</h4>
                                    <p className="author-role">{review.role}</p>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Reviews;
