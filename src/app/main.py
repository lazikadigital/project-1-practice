from .physics_utils import MassSpringSystem, spring_force


def main() -> None:
    sys = MassSpringSystem(m=1.0, k=10.0, c=0.5)
    print(sys.force(x=0.1, v=0.0))


if __name__ == "__main__":
    main()


# from app.physics_utils import MassSpringSystem, spring_force

# python -m venv venv  - Creates virtual environment
# .\venv\Scripts\Activate.ps1 - Activates virtual environment
# python -m src.app.main  - Starts the app
