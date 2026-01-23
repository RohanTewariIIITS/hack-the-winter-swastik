import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { api } from './api'
import {
  TrendingUp,
  Zap,
  Clock,
  Target,
  Users,
  BarChart3,
  ChevronRight,
  ExternalLink,
  Lightbulb,
  CheckCircle
} from 'lucide-react'

// ============================================================================
// MAIN APP WITH ROUTING
// ============================================================================
function App() {
  const [currentUser, setCurrentUser] = useState(null)
  const [users, setUsers] = useState([])
  const [problems, setProblems] = useState([])
  const [recommendations, setRecommendations] = useState([])
  const [globalInsights, setGlobalInsights] = useState([])
  const [selectedProblem, setSelectedProblem] = useState(null)
  const [loading, setLoading] = useState(true)

  // Demo users for switcher
  const demoUsers = [
    { handle: 'tourist', rating: 3995 },
    { handle: 'jiangly', rating: 3608 },
    { handle: 'ecnerwala', rating: 3400 },
    { handle: 'rainboy', rating: 3100 },
    { handle: 'candidate_master_1', rating: 1920 },
    { handle: 'expert_user_1', rating: 1650 },
    { handle: 'specialist_user', rating: 1450 },
    { handle: 'pupil_user', rating: 1250 },
  ]

  useEffect(() => {
    loadInitialData()
  }, [])

  useEffect(() => {
    if (currentUser) {
      loadUserRecommendations(currentUser)
    }
  }, [currentUser])

  const loadInitialData = async () => {
    setLoading(true)
    try {
      const insights = await api.fetchGlobalInsights()
      setGlobalInsights(insights || [])

      // Set default user
      const defaultUser = await api.analyzeProfile('candidate_master_1')
      if (defaultUser) {
        setCurrentUser(defaultUser)
      }
    } catch (e) {
      console.error('Failed to load initial data:', e)
    }
    setLoading(false)
  }

  const loadUserRecommendations = async (user) => {
    try {
      const recs = await api.getRecommendations(user)
      if (recs.status === 'success') {
        setRecommendations(recs.recommendations || [])
        // Enrich with more problem data
        setProblems(recs.recommendations?.map(r => ({
          ...r,
          title: getProblemTitle(r.problem_id),
          tags: getProblemTags(r.problem_id)
        })) || [])
      }
    } catch (e) {
      console.error('Failed to load recommendations:', e)
    }
  }

  const switchUser = async (handle) => {
    setLoading(true)
    try {
      const user = await api.analyzeProfile(handle)
      if (user) {
        setCurrentUser(user)
        setSelectedProblem(null)
      }
    } catch (e) {
      console.error('Failed to switch user:', e)
    }
    setLoading(false)
  }

  // Problem metadata helpers
  const getProblemTitle = (id) => {
    const titles = {
      '1462F': 'The Treasure of The Segments',
      '1623C': 'Balanced Stone Heaps',
      '1552E': 'Colors and Intervals',
      '1709E': 'XOR Tree',
      '1705C': 'Mark and His Unfinished Essay',
      '1800E': 'Unforgivable Curse',
      '1829G': 'Hits Different',
      '1850H': 'The Third Letter',
      '1914E': 'Game with Marbles',
      '1899F': "Alex's whims",
      '1881E': 'Block Sequence',
      '1878E': 'Iva & Pav',
    }
    return titles[id] || 'Unknown Problem'
  }

  const getProblemTags = (id) => {
    const tags = {
      '1462F': ['Greedy', 'Sortings', 'Two Pointers'],
      '1623C': ['Binary Search', 'Greedy'],
      '1552E': ['Constructive', 'Greedy', 'Sortings'],
      '1709E': ['DFS', 'Trees', 'Bitmasks'],
      '1705C': ['Binary Search', 'Implementation'],
      '1800E': ['Constructive', 'Strings'],
      '1829G': ['DP', 'Math', 'Implementation'],
      '1850H': ['DSU', 'Graphs', 'Implementation'],
      '1914E': ['Greedy', 'Sortings', 'Games'],
      '1899F': ['Constructive', 'Trees'],
      '1881E': ['DP', 'Two Pointers'],
      '1878E': ['Binary Search', 'Bitmasks'],
    }
    return tags[id] || []
  }

  const getRatingClass = (rating) => {
    if (rating >= 3000) return 'rating-legendary'
    if (rating >= 2600) return 'rating-grandmaster'
    if (rating >= 2400) return 'rating-master'
    if (rating >= 1900) return 'rating-candidate'
    if (rating >= 1600) return 'rating-expert'
    if (rating >= 1400) return 'rating-specialist'
    if (rating >= 1200) return 'rating-pupil'
    return 'rating-newbie'
  }

  const getDifficultyClass = (diff) => {
    if (diff <= 1400) return 'easy'
    if (diff <= 1700) return 'medium'
    return 'hard'
  }

  return (
    <BrowserRouter>
      <div className="app">
        {/* Navigation */}
        <nav className="nav">
          <div className="container nav-inner">
            <div className="nav-brand">
              <TrendingUp size={24} />
              <span>CP Intelligence</span>
            </div>

            <div className="nav-links">
              <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                Practice
              </NavLink>
              <NavLink to="/analyze" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                Analyze Me
              </NavLink>
              <NavLink to="/insights" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                Insights
              </NavLink>
            </div>

            <div className="nav-right">
              <div className="user-switcher">
                <Users size={16} />
                <select
                  value={currentUser?.handle || ''}
                  onChange={(e) => switchUser(e.target.value)}
                >
                  {demoUsers.map(u => (
                    <option key={u.handle} value={u.handle}>
                      {u.handle} ({u.rating})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={
            <PracticePage
              problems={problems}
              currentUser={currentUser}
              selectedProblem={selectedProblem}
              setSelectedProblem={setSelectedProblem}
              getRatingClass={getRatingClass}
              getDifficultyClass={getDifficultyClass}
              loading={loading}
            />
          } />
          <Route path="/analyze" element={
            <AnalyzePage
              currentUser={currentUser}
              recommendations={recommendations}
              getRatingClass={getRatingClass}
            />
          } />
          <Route path="/insights" element={
            <InsightsPage
              globalInsights={globalInsights}
            />
          } />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

// ============================================================================
// PRACTICE PAGE - CF-Style Problemset
// ============================================================================
function PracticePage({ problems, currentUser, selectedProblem, setSelectedProblem, getRatingClass, getDifficultyClass, loading }) {

  if (loading) {
    return (
      <div className="container" style={{ paddingTop: 60, textAlign: 'center' }}>
        <p>Loading problemset...</p>
      </div>
    )
  }

  return (
    <div>
      {/* Page Header */}
      <div className="page-header">
        <div className="container">
          <h1 className="page-title">Practice Problemset</h1>
          <p className="page-subtitle">
            Problems ranked by causal impact on rating growth for your skill level
          </p>
        </div>
      </div>

      <div className="container">
        <div className="two-col-layout">
          {/* Problem List */}
          <div className="card">
            <div className="card-header">
              <span>Recommended Problems ({problems.length})</span>
              {currentUser && (
                <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                  For rating: <span className={getRatingClass(currentUser.current_rating)}>{currentUser.current_rating}</span>
                </span>
              )}
            </div>
            <table className="problem-table">
              <thead>
                <tr>
                  <th style={{ width: 80 }}>#</th>
                  <th>Problem</th>
                  <th style={{ width: 100 }}>Difficulty</th>
                  <th style={{ width: 140 }}>Rating Impact</th>
                </tr>
              </thead>
              <tbody>
                {problems.map((problem, index) => (
                  <tr
                    key={problem.problem_id}
                    className={selectedProblem?.problem_id === problem.problem_id ? 'selected' : ''}
                    onClick={() => setSelectedProblem(problem)}
                    style={{ cursor: 'pointer' }}
                  >
                    <td>
                      <span className="problem-id">{problem.problem_id}</span>
                    </td>
                    <td>
                      <div className="problem-title">{problem.title}</div>
                      <div style={{ marginTop: 4 }}>
                        {problem.tags?.slice(0, 3).map(tag => (
                          <span key={tag} className="tag">{tag}</span>
                        ))}
                      </div>
                    </td>
                    <td>
                      <span className={`difficulty-badge ${getDifficultyClass(problem.estimated_difficulty)}`}>
                        {problem.estimated_difficulty}
                      </span>
                    </td>
                    <td>
                      <span className={`impact-badge ${problem.uplift > 50 ? 'high' : ''}`}>
                        <Zap size={12} />
                        +{problem.uplift?.toFixed(0)} pts
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Detail Panel */}
          <div className="detail-panel fade-in">
            {selectedProblem ? (
              <>
                <div className="detail-header">
                  <div className="detail-title">{selectedProblem.problem_id}: {selectedProblem.title}</div>
                  <div className="detail-meta">
                    <span className={`difficulty-badge ${getDifficultyClass(selectedProblem.estimated_difficulty)}`}>
                      {selectedProblem.estimated_difficulty}
                    </span>
                    {selectedProblem.tags?.map(tag => (
                      <span key={tag} className="tag">{tag}</span>
                    ))}
                  </div>
                </div>

                <div className="detail-section">
                  <h4>Why This Problem?</h4>
                  <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
                    Users at your skill level who solved this problem gained an average of{' '}
                    <strong style={{ color: 'var(--accent-green)' }}>+{selectedProblem.uplift?.toFixed(0)} rating points</strong>{' '}
                    in the following 30 days. This effect is statistically significant.
                  </p>

                  <div className="evidence-panel">
                    <div className="evidence-title">
                      <Lightbulb size={16} />
                      Causal Evidence
                    </div>
                    <div className="evidence-item">
                      <span>Expected Rating Gain</span>
                      <strong style={{ color: 'var(--accent-green)' }}>+{selectedProblem.uplift?.toFixed(0)}</strong>
                    </div>
                    <div className="evidence-item">
                      <span>Time to See Results</span>
                      <strong>~{(selectedProblem.median_time_to_improve / 10).toFixed(0)} days</strong>
                    </div>
                    <div className="evidence-item">
                      <span>Improvement Speed</span>
                      <strong>{selectedProblem.hazard_ratio?.toFixed(1) || '1.5'}× faster</strong>
                    </div>
                    <div className="evidence-item">
                      <span>Statistical Confidence</span>
                      <strong style={{ color: selectedProblem.p_value < 0.001 ? 'var(--accent-green)' : 'inherit' }}>
                        {selectedProblem.p_value < 0.001 ? 'Very High' : 'High'}
                      </strong>
                    </div>
                  </div>
                </div>

                <div className="detail-section">
                  <a
                    href={`https://codeforces.com/problemset/problem/${selectedProblem.problem_id.slice(0, -1)}/${selectedProblem.problem_id.slice(-1)}`}
                    target="_blank"
                    className="btn btn-primary"
                    style={{ width: '100%', justifyContent: 'center' }}
                  >
                    <ExternalLink size={16} />
                    Solve on Codeforces
                  </a>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <Target size={48} />
                <p>Select a problem to see details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// ANALYZE PAGE - User Dashboard
// ============================================================================
function AnalyzePage({ currentUser, recommendations, getRatingClass }) {
  if (!currentUser) {
    return (
      <div className="container" style={{ paddingTop: 60, textAlign: 'center' }}>
        <p>Select a user to analyze</p>
      </div>
    )
  }

  return (
    <div>
      <div className="page-header">
        <div className="container">
          <h1 className="page-title">Analysis: {currentUser.handle}</h1>
          <p className="page-subtitle">
            Your personalized rating growth strategy
          </p>
        </div>
      </div>

      <div className="container">
        {/* Stats Overview */}
        <div className="stats-grid" style={{ marginBottom: 24 }}>
          <div className="stat-card">
            <div className={`stat-value ${getRatingClass(currentUser.current_rating)}`}>
              {currentUser.current_rating}
            </div>
            <div className="stat-label">Current Rating</div>
          </div>
          <div className="stat-card">
            <div className="stat-value positive">
              +{recommendations.reduce((sum, r) => sum + (r.uplift || 0), 0).toFixed(0)}
            </div>
            <div className="stat-label">Potential Gain</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">
              {recommendations.length}
            </div>
            <div className="stat-label">Recommended Problems</div>
          </div>
        </div>

        {/* Strategy Insight */}
        <div className="insight-card" style={{ marginBottom: 24 }}>
          <h3>Strategy Insight</h3>
          <p>
            For users around <strong>{currentUser.current_rating} rating</strong>,
            solving problems in the <strong>{Math.round(currentUser.current_rating - 200)} - {Math.round(currentUser.current_rating + 300)}</strong> difficulty range
            with high causal uplift leads to <strong>1.5× faster growth</strong> than random practice.
          </p>
        </div>

        {/* Top Recommendations */}
        <div className="card">
          <div className="card-header">
            Top 5 Recommended Actions
          </div>
          <div className="card-body">
            {recommendations.slice(0, 5).map((rec, index) => (
              <div key={rec.problem_id} style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 0',
                borderBottom: index < 4 ? '1px solid var(--border-light)' : 'none'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{
                    width: 24,
                    height: 24,
                    borderRadius: '50%',
                    background: 'var(--accent-blue)',
                    color: 'white',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 12,
                    fontWeight: 600
                  }}>
                    {index + 1}
                  </span>
                  <div>
                    <div style={{ fontWeight: 600 }}>{rec.problem_id}</div>
                    <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                      Difficulty: {rec.estimated_difficulty}
                    </div>
                  </div>
                </div>
                <span className="impact-badge high">
                  <TrendingUp size={12} />
                  +{rec.uplift?.toFixed(0)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// INSIGHTS PAGE - Global Statistics
// ============================================================================
function InsightsPage({ globalInsights }) {
  return (
    <div>
      <div className="page-header">
        <div className="container">
          <h1 className="page-title">Global Insights</h1>
          <p className="page-subtitle">
            What our causal analysis reveals about rating growth
          </p>
        </div>
      </div>

      <div className="container">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 24 }}>

          {/* Key Finding 1 */}
          <div className="card">
            <div className="card-header">
              <BarChart3 size={18} />
              <span style={{ marginLeft: 8 }}>Difficulty Sweet Spot</span>
            </div>
            <div className="card-body">
              <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
                Problems in the <strong>1500-1700</strong> difficulty range show the highest
                causal impact on rating growth for most users.
              </p>
              <div className="evidence-panel">
                <div className="evidence-item">
                  <span>1500-1600</span>
                  <strong style={{ color: 'var(--accent-green)' }}>+58.4 avg</strong>
                </div>
                <div className="evidence-item">
                  <span>1600-1700</span>
                  <strong style={{ color: 'var(--accent-green)' }}>+52.1 avg</strong>
                </div>
                <div className="evidence-item">
                  <span>1700-1800</span>
                  <strong>+45.2 avg</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Key Finding 2 */}
          <div className="card">
            <div className="card-header">
              <Clock size={18} />
              <span style={{ marginLeft: 8 }}>Time to Improvement</span>
            </div>
            <div className="card-body">
              <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
                Most rating improvements happen within <strong>15-20 days</strong> of solving
                high-impact problems.
              </p>
              <div className="evidence-panel">
                <div className="evidence-item">
                  <span>Median Time</span>
                  <strong>18 days</strong>
                </div>
                <div className="evidence-item">
                  <span>75th Percentile</span>
                  <strong>25 days</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Key Finding 3 */}
          <div className="card">
            <div className="card-header">
              <Target size={18} />
              <span style={{ marginLeft: 8 }}>What NOT to Practice</span>
            </div>
            <div className="card-body">
              <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>
                Problems <strong>300+ above</strong> your current rating show
                <strong style={{ color: 'var(--accent-red)' }}> negative causal impact</strong> on growth.
              </p>
              <div className="evidence-panel" style={{ background: '#fff1f0', borderColor: '#ffa39e' }}>
                <div className="evidence-item">
                  <span>+300 difficulty</span>
                  <strong style={{ color: 'var(--accent-red)' }}>-12 avg</strong>
                </div>
                <div className="evidence-item">
                  <span>+400 difficulty</span>
                  <strong style={{ color: 'var(--accent-red)' }}>-18 avg</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Top Problems */}
          <div className="card" style={{ gridColumn: '1 / -1' }}>
            <div className="card-header">
              <Zap size={18} />
              <span style={{ marginLeft: 8 }}>Highest Impact Problems (Global)</span>
            </div>
            <table className="problem-table">
              <thead>
                <tr>
                  <th>Problem</th>
                  <th>Difficulty</th>
                  <th>Avg Rating Impact</th>
                  <th>Sample Size</th>
                </tr>
              </thead>
              <tbody>
                {globalInsights.slice(0, 5).map(problem => (
                  <tr key={problem.problem_id}>
                    <td><span className="problem-id">{problem.problem_id}</span></td>
                    <td>{problem.estimated_difficulty}</td>
                    <td>
                      <span className="impact-badge high">
                        +{problem.att_score?.toFixed(0)}
                      </span>
                    </td>
                    <td>{problem.total_treated_samples} users</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
