export interface UserProfile {
  name: string;
  email: string;
  phone: string;
  education_level: "High School" | "Bachelor's" | "Master's" | "PhD" | "Other";
  education_field: string;
  years_experience: number;
  skills: string[];
  interests: string[];
  certificates: string[];
  profile_summary: string;
}

export interface Recommendation {
  title: string;
  platform: string;
  provider: string;
  difficulty: string;
  rating?: number | string;
  url: string;
  description: string;
  relevance_score?: number;
  explanation?: string;
}

export interface UploadResponse {
  status: "success" | "error";
  user_profile: UserProfile;
  recommendations: Recommendation[];
  error?: string;
}
