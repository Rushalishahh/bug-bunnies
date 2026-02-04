import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
    Users, Activity, Brain, Calendar, Search,
    Bell, Settings, LogOut, ClipboardList, Plus,
    MoreVertical, Mail, Phone, Filter
} from 'lucide-react';
import './PatientList.css';

const patientsData = [
    { id: 1, name: 'Arjun Verma', age: 12, condition: 'Articulation', status: 'Active', sessions: 15, lastVisit: '2026-02-01' },
    { id: 2, name: 'Isha Khan', age: 8, condition: 'Fluency', status: 'Active', sessions: 22, lastVisit: '2026-01-30' },
    { id: 3, name: 'Rayan Malhotra', age: 10, condition: 'Language Delay', status: 'Needs Attention', sessions: 8, lastVisit: '2026-01-15' },
    { id: 4, name: 'Priya Sharma', age: 15, condition: 'Voice Disorder', status: 'Recovered', sessions: 30, lastVisit: '2026-01-20' },
    { id: 5, name: 'Kabir Singh', age: 7, condition: 'Stuttering', status: 'Active', sessions: 12, lastVisit: '2026-02-02' },
    { id: 6, name: 'Ananya Gupta', age: 11, condition: 'Cognitive Communication', status: 'Active', sessions: 18, lastVisit: '2026-01-28' },
    { id: 7, name: 'Dev Patel', age: 9, condition: 'Apraxia', status: 'Needs Attention', sessions: 5, lastVisit: '2026-01-10' },
];

const PatientList = () => {
    const location = useLocation();
    const { name = 'Dr. Rajesh Gupta', clinic = 'Arogya Speech Center' } = location.state || {};

    return (
        <div className="dashboard-container">
            {/* Sidebar */}
            <aside className="dashboard-sidebar">
                <div className="sidebar-logo">S<span>peechSync</span></div>
                <nav className="sidebar-nav">
                    <Link to="/dashboard/therapist" className="nav-item"><ClipboardList size={20} /> Overview</Link>
                    <Link to="/dashboard/therapist/patients" className="nav-item active"><Users size={20} /> Patients</Link>
                    <a href="#" className="nav-item"><Calendar size={20} /> Schedule</a>
                    <a href="#" className="nav-item"><Brain size={20} /> AI Analysis</a>
                    <a href="#" className="nav-item"><Bell size={20} /> Notifications</a>
                    <div className="nav-divider"></div>
                    <a href="#" className="nav-item"><Settings size={20} /> Settings</a>
                    <Link to="/" className="nav-item logout"><LogOut size={20} /> Logout</Link>
                </nav>
            </aside>

            {/* Main Content */}
            <main className="dashboard-main">
                {/* Header */}
                <header className="dashboard-header">
                    <div className="header-search">
                        <Search size={18} />
                        <input type="text" placeholder="Search patients by name or ID..." />
                    </div>
                    <div className="header-profile">
                        <div className="profile-info">
                            <span className="profile-name">Hello, {name}</span>
                            <span className="profile-clinic">{clinic}</span>
                        </div>
                        <div className="profile-avatar">{name.includes(' ') ? name.split(' ')[1].charAt(0) : name.charAt(0)}</div>
                    </div>
                </header>

                <div className="dashboard-content">
                    <div className="page-header">
                        <div>
                            <h2>Patient Management</h2>
                            <p>You have {patientsData.length} total patients registered.</p>
                        </div>
                        <div className="header-actions">
                            <button className="secondary-btn"><Filter size={18} /> Filter</button>
                            <Link to="/dashboard/therapist/patients/add" className="add-patient-btn"><Plus size={18} /> Add New Patient</Link>
                        </div>
                    </div>

                    <div className="patients-table-container">
                        <table className="patients-table">
                            <thead>
                                <tr>
                                    <th>Patient Name</th>
                                    <th>Age</th>
                                    <th>Condition</th>
                                    <th>Total Sessions</th>
                                    <th>Last Visit</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {patientsData.map((patient) => (
                                    <tr key={patient.id}>
                                        <td>
                                            <div className="patient-name-cell">
                                                <div className="patient-initials">{patient.name.split(' ').map(n => n[0]).join('')}</div>
                                                <div>
                                                    <div className="name-bold">{patient.name}</div>
                                                    <div className="id-sub">ID: #SS-{1000 + patient.id}</div>
                                                </div>
                                            </div>
                                        </td>
                                        <td>{patient.age} years</td>
                                        <td>{patient.condition}</td>
                                        <td>{patient.sessions}</td>
                                        <td>{patient.lastVisit}</td>
                                        <td>
                                            <span className={`status-tag ${patient.status.toLowerCase().replace(' ', '-')}`}>
                                                {patient.status}
                                            </span>
                                        </td>
                                        <td>
                                            <button className="icon-btn"><MoreVertical size={18} /></button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default PatientList;
