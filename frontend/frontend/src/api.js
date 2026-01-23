// Use environment variable for production, fallback to localhost for dev
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const api = {
  fetchGlobalInsights: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/global-insights`);
      if (!response.ok) throw new Error("Failed to fetch global insights");
      return await response.json();
    } catch (error) {
      console.error(error);
      return [];
    }
  },

  fetchProblemDetails: async (problemId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/problem-details/${problemId}`);
      if (!response.ok) throw new Error("Failed to fetch problem details");
      return await response.json();
    } catch (error) {
      console.error(error);
      return null;
    }
  },

  analyzeProfile: async (handle) => {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze-profile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handle }),
      });
      if (!response.ok) throw new Error("User not found or CF API error");
      return await response.json();
    } catch (error) {
      console.error(error);
      return null;
    }
  },

  getRecommendations: async (userProfile) => {
    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(userProfile),
      });
      if (!response.ok) throw new Error("Failed to get recommendations");
      return await response.json();
    } catch (error) {
      console.error(error);
      return { status: "error", message: error.message };
    }
  },
};
