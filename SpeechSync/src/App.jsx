import React from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import PortalSelection from './pages/PortalSelection';
import TherapistSignup from './pages/TherapistSignup';
import TherapistDashboard from './pages/TherapistDashboard';
import PatientList from './pages/PatientList';
import AddPatient from './pages/AddPatient';
import PatientLogin from './pages/PatientLogin';
import TherapistLogin from './pages/TherapistLogin';
import PatientDashboard from './pages/PatientDashboard';
import './App.css';

function App() {
  return (
    <div className="app">
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/get-started" element={<PortalSelection />} />
        <Route path="/portal" element={<PortalSelection />} />
        <Route path="/signup/therapist" element={<TherapistSignup />} />
        <Route path="/dashboard/therapist" element={<TherapistDashboard />} />
        <Route path="/dashboard/therapist/patients" element={<PatientList />} />
        <Route path="/dashboard/therapist/patients/add" element={<AddPatient />} />
        <Route path="/login/patient" element={<PatientLogin />} />
        <Route path="/login/therapist" element={<TherapistLogin />} />
        <Route path="/dashboard/patient" element={<PatientDashboard />} />
      </Routes>
    </div>
  );
}

export default App;
