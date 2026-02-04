import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav className={`navbar ${isScrolled ? 'scrolled' : ''}`}>
      <div className="container nav-content">
        <Link to="/" className="logo">
          <span className="logo-icon">S</span>
          SpeechSync
        </Link>
        <div className="nav-links">
          <a href="#platform" className="nav-link">Platform</a>
          <a href="#how-it-works" className="nav-link">How It Works</a>
          <a href="#reviews" className="nav-link">Reviews</a>
          <Link to="/portal" className="nav-link login-link">Login</Link>
          <Link to="/get-started" className="btn btn-primary">Get Started</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
