from CA import run_ca
from BrakedownCA import run_brakedownca
from imageextract import run_image_extraction
from ucagenerate import run_uca_generation
from ucasort import run_uca_sort

def main():
    print("\n🚀 Starting the LLM-based Safety Analysis Tool...\n")

    print("📘 Step 1: Creating control knowledge...")
    run_ca()

    print("🧠 Step 2: Mapping user input to control actions...")
    run_brakedownca()

    print("🖼️ Step 3: Extracting and analyzing scene images...")
    run_image_extraction()

    print("📌 Step 4: Generating UCAs and linking them to hazards...")
    run_uca_generation()

    print("📊 Step 5: Scoring UCAs and summarizing contributions...")
    run_uca_sort("ucasort_output.json")

    print("\n✅ Safety analysis completed successfully!")

if __name__ == "__main__":
    main()
