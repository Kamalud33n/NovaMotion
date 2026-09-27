"""Role and account-status constants — the one place these strings live."""

ROLE_ADMIN     = "admin"
ROLE_DOCTOR    = "doctor"
ROLE_THERAPIST = "therapist"

ALL_ROLES = (ROLE_ADMIN, ROLE_DOCTOR, ROLE_THERAPIST)

# Doctors and therapists share the same (clinical) pages and permissions.
CLINICAL_ROLES = (ROLE_DOCTOR, ROLE_THERAPIST)

# Roles a person may pick for themselves on the register form.
# Admin is never self-registered — it is seeded from .env or created by an admin.
SELF_REGISTER_ROLES = CLINICAL_ROLES

STATUS_PENDING   = "pending"     # registered, waiting for admin approval
STATUS_APPROVED  = "approved"    # can log in
STATUS_REJECTED  = "rejected"    # admin declined the registration
STATUS_SUSPENDED = "suspended"   # admin disabled a previously approved account

ALL_STATUSES = (STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED, STATUS_SUSPENDED)


def home_for_role(role: str) -> str:
    """Landing page after login / when a role hits a page it may not open."""
    return "/admin" if role == ROLE_ADMIN else "/dashboard"
