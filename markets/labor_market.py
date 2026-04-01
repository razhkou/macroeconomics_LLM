import numpy as np
from typing import Dict, List, Tuple, Any


class LaborMarket:
    def __init__(self):
        self.vacancies = []
        self.applications = []
        self.matches = {}
        self.hiring = {}
        self.wages = {}

    def reset(self):
        self.vacancies = []
        self.applications = []
        self.matches = {}
        self.hiring = {}
        self.wages = {}

    def add_vacancy(self, firm_id: str, wage: float, required_skills: List[float], count: int):
        self.vacancies.append((firm_id, wage, required_skills, count))

    def add_application(self, hh_id: str, work_ratio: float, desired_wage: float, skills: List[float]):
        self.applications.append((hh_id, work_ratio, desired_wage, skills))

    def match(self):
        self.matches = {fid: [] for fid, _, _, _ in self.vacancies}
        self.hiring = {fid: 0 for fid, _, _, _ in self.vacancies}
        self.wages = {}

        vacancies_sorted = sorted(self.vacancies, key=lambda x: -x[1])
        applications_sorted = sorted(self.applications, key=lambda x: -sum(x[3]))

        for firm_id, wage, req_skills, count in vacancies_sorted:
            hired = 0
            for i, (hh_id, work_ratio, desired_wage, skills) in enumerate(applications_sorted[:]):
                if hired >= count:
                    break
                compatibility = np.dot(skills, req_skills) / (np.linalg.norm(skills) * np.linalg.norm(req_skills) + 1e-8)
                if compatibility > 0.5 and desired_wage <= wage:
                    self.matches[firm_id].append(hh_id)
                    self.hiring[firm_id] += 1
                    self.wages[hh_id] = wage
                    hired += 1
                    applications_sorted.pop(i)

    def get_employees(self, firm_id: str) -> List[str]:
        return self.matches.get(firm_id, [])

    def get_wage(self, hh_id: str) -> float:
        return self.wages.get(hh_id, 0)