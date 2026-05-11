import numpy as np
from typing import Dict, List, Tuple, Any


class LaborMarket:
    def __init__(self):
        self.vacancies = []          # (firm_id, wage, required_skills, count)
        self.applications = []       # (hh_id, available_workers, desired_wage, skills)
        self.matches = {}            # firm_id -> list of (hh_id, workers_hired, wage)
        self.hiring = {}             # firm_id -> total workers hired
        self.wages = {}              # hh_id -> list of (wage, workers_assigned)

    def reset(self):
        self.vacancies = []
        self.applications = []
        self.matches = {}
        self.hiring = {}
        self.wages = {}

    def add_vacancy(self, firm_id: str, wage: float, required_skills: List[float], count: int):
        self.vacancies.append((firm_id, wage, required_skills, count))

    def add_application(self, hh_id: str, workers: float, desired_wage: float, skills: List[float]):
        if workers > 0:
            self.applications.append((hh_id, workers, desired_wage, skills))

    def match(self):
        vacancies_sorted = sorted(self.vacancies, key=lambda x: -x[1])
        applications_sorted = sorted(self.applications, key=lambda x: -sum(x[3]))

        self.matches = {}
        self.hiring = {fid: 0 for fid, _, _, _ in self.vacancies}
        self.wages = {}

        for firm_id, wage, req_skills, vacancy_count in vacancies_sorted:
            hired = 0
            i = 0
            while i < len(applications_sorted) and hired < vacancy_count:
                hh_id, workers_available, desired_wage, skills = applications_sorted[i]
                compatibility = np.dot(skills, req_skills) / (np.linalg.norm(skills) * np.linalg.norm(req_skills) + 1e-8)
                if compatibility > 0.5 and desired_wage <= wage:
                    to_hire = min(workers_available, vacancy_count - hired)
                    if firm_id not in self.matches:
                        self.matches[firm_id] = []
                    self.matches[firm_id].append((hh_id, to_hire, wage))
                    self.hiring[firm_id] += to_hire
                    if hh_id not in self.wages:
                        self.wages[hh_id] = []
                    self.wages[hh_id].append((wage, to_hire))
                    workers_available -= to_hire
                    hired += to_hire
                    if workers_available <= 0:
                        applications_sorted.pop(i)
                        continue
                    else:
                        applications_sorted[i] = (hh_id, workers_available, desired_wage, skills)
                i += 1