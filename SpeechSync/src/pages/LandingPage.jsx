import React from 'react';
import Navbar from '../components/Navbar';
import Hero from '../components/Hero';
import Stats from '../components/Stats';
import Reviews from '../components/Reviews';
import FinalCTA from '../components/FinalCTA';
import Footer from '../components/Footer';
import backgroundVideo from '../components/watermarked_preview.mp4';

const LandingPage = () => {
    return (
        <>
            <video autoPlay loop muted playsInline className="background-video">
                <source src={backgroundVideo} type="video/mp4" />
            </video>
            <div className="video-overlay"></div>

            <Navbar />
            <main>
                <Hero />
                <Stats />
                <Reviews />
                <FinalCTA />
            </main>
            <Footer />
        </>
    );
};

export default LandingPage;
