import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
    Users, Activity, Brain, TrendingUp, Calendar, Search,
    Bell, Settings, LogOut, ChevronRight, MessageSquare, ClipboardList, Plus,
    Hand
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell, Legend
} from 'recharts';
import './TherapistDashboard.css';

const performanceData = [
    { name: 'Mon', sessions: 12, improvement: 65 },
    { name: 'Tue', sessions: 18, improvement: 72 },
    { name: 'Wed', sessions: 15, improvement: 68 },
    { name: 'Thu', sessions: 22, improvement: 85 },
    { name: 'Fri', sessions: 20, improvement: 78 },
    { name: 'Sat', sessions: 10, improvement: 90 },
    { name: 'Sun', sessions: 5, improvement: 92 },
];

const exerciseData = [
    { name: 'Articulation', count: 45 },
    { name: 'Fluency', count: 32 },
    { name: 'Voice', count: 28 },
    { name: 'Language', count: 38 },
    { name: 'Cognition', count: 15 },
];

const statusData = [
    { name: 'Active', value: 85, color: '#0d9488' },
    { name: 'Recovered', value: 30, color: '#059669' },
    { name: 'Needs Attention', value: 9, color: '#ef4444' },
];

const TherapistDashboard = () => {
    const location = useLocation();
    const { name = 'Dr. Rajesh Gupta', clinic = 'Arogya Speech Center', isDemo = false } = location.state || {};

    // Mock data for demo, otherwise empty
    const currentPerformanceData = isDemo ? performanceData : [];
    const currentExerciseData = isDemo ? exerciseData : [];
    const currentStatusData = isDemo ? statusData : [];
    const totalPatients = isDemo ? "124" : "0";
    const activeSessions = isDemo ? "48" : "0";
    const avgImprovement = isDemo ? "76%" : "0%";
    const scheduledSessions = isDemo ? 8 : 0;

    const nameParts = name.split(' ');
    const greetingName = (nameParts.length > 1 && ['Dr.', 'Dr', 'Mr.', 'Mr', 'Ms.', 'Ms', 'Mrs.', 'Mrs'].includes(nameParts[0]))
        ? `${nameParts[0]} ${nameParts[1]}`
        : nameParts[0];

    return (
        <div className="dashboard-container">
            {/* Sidebar */}
            <aside className="dashboard-sidebar">
                <div className="sidebar-logo">S<span>peechSync</span></div>
                <nav className="sidebar-nav">
                    <Link to="/dashboard/therapist" className="nav-item active"><ClipboardList size={20} /> Overview</Link>
                    {isDemo && (
                        <>
                            <Link to="/dashboard/therapist/patients" className="nav-item"><Users size={20} /> Patients</Link>
                            <a href="#" className="nav-item"><Calendar size={20} /> Schedule</a>
                            <a href="#" className="nav-item"><Brain size={20} /> AI Analysis</a>
                            <a href="#" className="nav-item"><Bell size={20} /> Notifications</a>
                        </>
                    )}
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
                        <input type="text" placeholder="Search patients, records..." />
                    </div>
                    <div className="header-profile">
                        <div className="profile-info">
                            <span className="profile-name">Hello, {name}</span>
                            <span className="profile-clinic">{clinic}</span>
                        </div>
                        <div className="profile-avatar">{name.includes(' ') ? name.split(' ')[1].charAt(0) : name.charAt(0)}</div>
                    </div>
                </header>

                {/* Overview Wrapper */}
                <div className="dashboard-content">
                    <div className="content-intro">
                        <div className="intro-text">
                            <h2>Welcome {isDemo ? 'back' : ''}, {greetingName}</h2>
                            <p>{isDemo ? `Everything looks good. You have ${scheduledSessions} sessions scheduled for today.` : 'Welcome to your new dashboard. Start by adding your first patient.'}</p>
                        </div>
                        <Link to="/dashboard/therapist/patients/add" className="add-patient-btn">
                            <Plus size={18} /> Add New Patient
                        </Link>
                    </div>

                    {/* Stats Bar */}
                    <div className="dashboard-stats-grid">
                        <div className="stat-card-mini">
                            <div className="stat-info">
                                <span className="stat-label">Total Patients</span>
                                <span className="stat-value">{totalPatients}</span>
                                {isDemo && <span className="stat-change positive">+12% from last month</span>}
                            </div>
                            <div className="stat-icon-wrapper blue"><Users size={24} /></div>
                        </div>
                        <div className="stat-card-mini">
                            <div className="stat-info">
                                <span className="stat-label">Active Sessions</span>
                                <span className="stat-value">{activeSessions}</span>
                                {isDemo && <span className="stat-change">6 pending review</span>}
                            </div>
                            <div className="stat-icon-wrapper teal"><Activity size={24} /></div>
                        </div>
                        <div className="stat-card-mini">
                            <div className="stat-info">
                                <span className="stat-label">Avg. Improvement</span>
                                <span className="stat-value">{avgImprovement}</span>
                                {isDemo && <span className="stat-change positive">+5.4% improvement</span>}
                            </div>
                            <div className="stat-icon-wrapper green"><TrendingUp size={24} /></div>
                        </div>
                    </div>

                    {/* Analytics Section - Row 1 */}
                    {isDemo && (
                        <>
                            <div className="analytics-section">
                                <div className="chart-container main-chart">
                                    <div className="chart-header">
                                        <h3>Weekly Therapy Performance</h3>
                                        <div className="chart-legend">
                                            <span className="legend-dot sessions"></span> Sessions
                                        </div>
                                    </div>
                                    <div className="chart-wrapper">
                                        <ResponsiveContainer width="100%" height={300}>
                                            <AreaChart data={currentPerformanceData}>
                                                <defs>
                                                    <linearGradient id="colorSessions" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3} />
                                                        <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                                <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                                <Tooltip
                                                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                                                />
                                                <Area
                                                    type="monotone"
                                                    dataKey="sessions"
                                                    stroke="var(--primary)"
                                                    fillOpacity={1}
                                                    fill="url(#colorSessions)"
                                                    strokeWidth={3}
                                                />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                <div className="chart-container secondary-chart">
                                    <div className="chart-header">
                                        <h3>Exercise Distribution</h3>
                                    </div>
                                    <div className="chart-wrapper">
                                        <ResponsiveContainer width="100%" height={300}>
                                            <BarChart data={currentExerciseData} layout="vertical">
                                                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                                                <XAxis type="number" hide />
                                                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{ fill: '#475569', fontSize: 12 }} width={80} />
                                                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }} />
                                                <Bar dataKey="count" fill="var(--primary)" radius={[0, 10, 10, 0]} barSize={20} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>

                            {/* Analytics Section - Row 2 */}
                            <div className="analytics-section bottom-row">
                                <div className="chart-container">
                                    <div className="chart-header">
                                        <h3>Patient Status</h3>
                                    </div>
                                    <div className="chart-wrapper pie-wrapper">
                                        <ResponsiveContainer width="100%" height={250}>
                                            <PieChart>
                                                <Pie
                                                    data={currentStatusData}
                                                    cx="50%"
                                                    cy="50%"
                                                    innerRadius={60}
                                                    outerRadius={80}
                                                    paddingAngle={8}
                                                    dataKey="value"
                                                >
                                                    {currentStatusData.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                                    ))}
                                                </Pie>
                                                <Tooltip />
                                                <Legend verticalAlign="bottom" height={36} />
                                            </PieChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                <div className="ai-summary-container">
                                    <div className="summary-header">
                                        <h3><Brain size={18} /> AI Progress Summaries</h3>
                                        <Link to="#">View All <ChevronRight size={16} /></Link>
                                    </div>
                                    <div className="summary-list">
                                        <div className="summary-item">
                                            <div className="summary-patient-avatar">AV</div>
                                            <div className="summary-details">
                                                <p className="summary-text"><strong>Arjun V.</strong> showed significant progress in articulation of 's' sounds. Recommend increasing difficulty.</p>
                                                <span className="summary-tag high">High Improvement</span>
                                            </div>
                                        </div>
                                        <div className="summary-item">
                                            <div className="summary-patient-avatar">IK</div>
                                            <div className="summary-details">
                                                <p className="summary-text"><strong>Isha K.</strong> completed 5/5 home exercises. Consistency is improving.</p>
                                                <span className="summary-tag med">Consistent</span>
                                            </div>
                                        </div>
                                        <div className="summary-item">
                                            <div className="summary-patient-avatar">RM</div>
                                            <div className="summary-details">
                                                <p className="summary-text"><strong>Rayan M.</strong> missed 2 sessions. Consider scheduling a follow-up call.</p>
                                                <span className="summary-tag low">Needs Attention</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}

                    {!isDemo && (
                        <div className="empty-state-welcome">
                            <div className="welcome-card">
                                <div className="welcome-icon"><Hand size={48} strokeWidth={1.5} /></div>
                                <h3>Let's get started with your first patient</h3>
                                <p>To start tracking progress and using AI analytics, you need to add your patients and record their first baseline session.</p>
                                <Link to="/dashboard/therapist/patients/add" className="btn btn-primary add-first-patient">
                                    <Plus size={20} /> Add First Patient
                                </Link>
                            </div>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default TherapistDashboard;
