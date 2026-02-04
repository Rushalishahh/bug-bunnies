import React, { useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
    User, Mic, Play, Calendar, Award,
    TrendingUp, Activity, MessageSquare,
    Brain, LogOut, Bell, Settings, ArrowRight,
    Shield, Heart, CheckCircle2, Flame, X, FileText
} from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import './PatientDashboard.css';

const activityData = [
    { day: 'Mon', completion: 80 },
    { day: 'Tue', completion: 65 },
    { day: 'Wed', completion: 90 },
    { day: 'Thu', completion: 70 },
    { day: 'Fri', completion: 85 },
    { day: 'Sat', completion: 100 },
    { day: 'Sun', completion: 40 },
];

const PatientDashboard = () => {
    const location = useLocation();
    const { loginId = 'SS-724915', name = 'Arjun Verma' } = location.state || {};
    const [isRecordingBaseline, setIsRecordingBaseline] = useState(false);
    const [baselineStep, setBaselineStep] = useState(0);
    const [showReport, setShowReport] = useState(false);

    const handleDownloadPDF = () => {
        window.print();
    };

    const baselineTasks = [
        "Please say 'Ahhhhh' for as long as you can.",
        "Repeat the sentence: 'The quick brown fox jumps over the lazy dog.'",
        "Count from 1 to 10 slowly.",
        "Describe your favorite food in 3 sentences."
    ];

    const handleBaselineNext = () => {
        if (baselineStep < baselineTasks.length - 1) {
            setBaselineStep(prev => prev + 1);
        } else {
            setIsRecordingBaseline(false);
            setBaselineStep(0);
            alert("Baseline recording completed! Our ML system is now extracting features.");
        }
    };

    return (
        <div className="patient-dashboard">
            {/* Sidebar */}
            <aside className="p-sidebar">
                <div className="p-logo">S<span>peechSync</span></div>
                <nav className="p-nav">
                    <a href="#" className="p-nav-item active"><Activity size={20} /> Dashboard</a>
                    <a href="#" className="p-nav-item"><Calendar size={20} /> My Plan</a>
                    <a href="#" className="p-nav-item"><Award size={20} /> Achievements</a>
                    <a href="#" className="p-nav-item"><MessageSquare size={20} /> Messages</a>
                    <div className="p-nav-divider"></div>
                    <a href="#" className="p-nav-item"><Settings size={20} /> Settings</a>
                    <Link to="/" className="p-nav-item logout"><LogOut size={20} /> Logout</Link>
                </nav>
            </aside>

            {/* Main Content */}
            <main className="p-main">
                {/* Header */}
                <header className="p-header">
                    <div className="p-welcome">
                        <h1>Hi, {name}!</h1>
                        <p>Total Practice Streak: <strong>12 Days</strong> <Flame size={18} className="streak-icon" style={{ display: 'inline', verticalAlign: 'middle', color: '#f59e0b' }} /></p>
                    </div>
                    <div className="p-header-actions">
                        <button className="p-icon-btn"><Bell size={20} /></button>
                        <div className="p-profile">
                            <div className="p-avatar">AV</div>
                        </div>
                    </div>
                </header>

                <div className="p-content">
                    {/* Top Row: Doctor Info & Quick Stats */}
                    <div className="p-info-grid">
                        <div className="p-doctor-card">
                            <div className="p-doctor-avatar">RG</div>
                            <div className="p-doctor-details">
                                <span className="p-label">Assigned Specialist</span>
                                <h3 className="p-doctor-name">Dr. Rajesh Gupta</h3>
                                <p className="p-clinic">Arogya Speech Center</p>
                            </div>
                            <button className="p-contact-btn">Message</button>
                        </div>
                        <div className="p-stat-card">
                            <div className="p-stat-icon purple"><TrendingUp size={24} /></div>
                            <div className="p-stat-info">
                                <span className="p-label">Progress Score</span>
                                <h3 className="p-value">72%</h3>
                                <span className="p-trend positive">+8% this week</span>
                            </div>
                        </div>
                        <div className="p-stat-card">
                            <div className="p-stat-icon orange"><Mic size={24} /></div>
                            <div className="p-stat-info">
                                <span className="p-label">Minutes Practiced</span>
                                <h3 className="p-value">340m</h3>
                                <span className="p-trend">50m left for goal</span>
                            </div>
                        </div>
                    </div>

                    {/* Middle Section: Baseline & Plan */}
                    <div className="p-work-grid">
                        {/* Baseline Recording Card */}
                        <div className={`p-card p-baseline-card ${isRecordingBaseline ? 'active-recording' : ''}`}>
                            <div className="p-card-header">
                                <h3><Shield size={20} /> Baseline Speech Assessment</h3>
                                {!isRecordingBaseline && <span className="p-badge pending">Required</span>}
                            </div>
                            {!isRecordingBaseline ? (
                                <div className="p-baseline-intro">
                                    <p>Help us understand your voice better. Complete a one-time baseline recording to set personalized goals.</p>
                                    <button onClick={() => setIsRecordingBaseline(true)} className="p-action-btn">
                                        Start Baseline Test <Mic size={18} />
                                    </button>
                                </div>
                            ) : (
                                <div className="p-baseline-active">
                                    <div className="p-task-box">
                                        <span className="p-task-num">Task {baselineStep + 1} of 4</span>
                                        <p className="p-task-text">{baselineTasks[baselineStep]}</p>
                                    </div>
                                    <div className="p-recording-ui">
                                        <div className="p-wave-animation">
                                            <span></span><span></span><span></span><span></span><span></span>
                                        </div>
                                        <button onClick={handleBaselineNext} className="p-record-finish-btn">
                                            {baselineStep === 3 ? 'Finish Assessment' : 'Next Task'}
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Daily Therapy Plan */}
                        <div className="p-card p-plan-card">
                            <div className="p-card-header">
                                <h3><Heart size={20} /> Today's Therapy Plan</h3>
                                <span className="p-date">Feb 4, 2026</span>
                            </div>
                            <div className="p-plan-list">
                                <div className="p-plan-item completed">
                                    <div className="p-plan-status"><CheckCircle2 size={18} /></div>
                                    <div className="p-plan-info">
                                        <h4>Warm-up: Lip Trills</h4>
                                        <p>3 minutes • Vocal focus</p>
                                    </div>
                                    <button className="p-play-btn"><Play size={16} /></button>
                                </div>
                                <div className="p-plan-item">
                                    <div className="p-plan-status"><div className="p-dot"></div></div>
                                    <div className="p-plan-info">
                                        <h4>'S' Sound Articulation</h4>
                                        <p>10 minutes • 15 repetitions</p>
                                    </div>
                                    <button className="p-action-btn small">Start</button>
                                </div>
                                <div className="p-plan-item">
                                    <div className="p-plan-status"><div className="p-dot"></div></div>
                                    <div className="p-plan-info">
                                        <h4>Narrative Description</h4>
                                        <p>5 minutes • Fluency focus</p>
                                    </div>
                                    <button className="p-action-btn small">Start</button>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Bottom Section: Analysis & Insights */}
                    <div className="p-analysis-grid">
                        <div className="p-card p-chart-card">
                            <div className="p-card-header">
                                <h3>Practice Consistency</h3>
                            </div>
                            <div className="p-chart-wrapper">
                                <ResponsiveContainer width="100%" height={250}>
                                    <AreaChart data={activityData}>
                                        <defs>
                                            <linearGradient id="colorComp" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                        <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                        <YAxis hide />
                                        <Tooltip
                                            contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' }}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="completion"
                                            stroke="var(--primary)"
                                            fillOpacity={1}
                                            fill="url(#colorComp)"
                                            strokeWidth={3}
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="p-card p-insight-card">
                            <div className="p-card-header">
                                <h3><Brain size={20} /> AI Insights</h3>
                            </div>
                            <div className="p-insight-list">
                                <div className="p-insight-item">
                                    <div className="p-insight-icon"><Brain size={18} /></div>
                                    <div className="p-insight-content">
                                        <p>Your <strong>clarity score</strong> improved by 15% when practicing during morning sessions.</p>
                                    </div>
                                </div>
                                <div className="p-insight-item">
                                    <div className="p-insight-icon"><TrendingUp size={18} /></div>
                                    <div className="p-insight-content">
                                        <p>Great job! You are hitting your 'S' sounds correctly 8/10 times now.</p>
                                    </div>
                                </div>
                            </div>
                            <button className="p-full-report-btn" onClick={() => setShowReport(true)}>View Full AI Analysis <ArrowRight size={16} /></button>
                        </div>
                    </div>
                </div>
            </main>

            {/* AI Report Modal */}
            {showReport && (
                <div className="report-overlay" onClick={() => setShowReport(false)}>
                    <div className="report-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="report-header">
                            <div className="report-title">
                                <Brain size={24} className="text-primary" />
                                <div>
                                    <h2>AI Speech Analysis Report</h2>
                                    <p>Patient ID: {loginId} | Generated on: {new Date().toLocaleDateString()}</p>
                                </div>
                            </div>
                            <div className="report-actions no-print">
                                <button className="btn-icon" onClick={handleDownloadPDF} title="Download PDF"><FileText size={20} /></button>
                                <button className="btn-icon" onClick={() => setShowReport(false)} title="Close"><X size={20} /></button>
                            </div>
                        </div>

                        <div className="report-body">
                            <div className="report-section">
                                <h3>Executive Summary</h3>
                                <p>Overall speech clarity has shown a <strong>15% improvement</strong> over the last 14 days. The patient is responding well to 'S' sound articulation exercises with higher consistency in morning sessions.</p>
                            </div>

                            <div className="report-grid">
                                <div className="report-card">
                                    <h4>Accuracy Metrics</h4>
                                    <div className="metric-row">
                                        <span>Articulation</span>
                                        <div className="progress-bar"><div className="progress" style={{ width: '82%' }}></div></div>
                                        <span>82%</span>
                                    </div>
                                    <div className="metric-row">
                                        <span>Fluency</span>
                                        <div className="progress-bar"><div className="progress" style={{ width: '65%' }}></div></div>
                                        <span>65%</span>
                                    </div>
                                    <div className="metric-row">
                                        <span>Vocal Range</span>
                                        <div className="progress-bar"><div className="progress" style={{ width: '78%' }}></div></div>
                                        <span>78%</span>
                                    </div>
                                </div>

                                <div className="report-card">
                                    <h4>Sound Processing Breakdown</h4>
                                    <ul className="sound-list">
                                        <li><strong>'S' Sounds:</strong> Significant improvement (80% accuracy)</li>
                                        <li><strong>'R' Sounds:</strong> Requires focus (45% accuracy)</li>
                                        <li><strong>'TH' Sounds:</strong> Developing consistency (60% accuracy)</li>
                                    </ul>
                                </div>
                            </div>

                            <div className="report-section recommendations">
                                <h3>AI-Generated Recommendations</h3>
                                <ul>
                                    <li>Increase repetition of 'S' articulation exercises to 20 reps per session.</li>
                                    <li>Focus on 'R' sound lip placement during the afternoon warm-ups.</li>
                                    <li>Maintain the current 12-day practice streak to solidify progress.</li>
                                </ul>
                            </div>

                            <div className="report-footer">
                                <div className="footer-notes">
                                    <p>Note: This is an AI-assisted report. Please consult your specialist, Dr. Rajesh Gupta, for medical interpretation.</p>
                                </div>
                                <div className="report-branding">
                                    <div className="p-logo">S<span>peechSync</span></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default PatientDashboard;
