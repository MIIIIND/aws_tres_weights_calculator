
#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Dict, List, Tuple
import boto3
import botocore
import numpy as np

REGION_TO_LOCATION = {
    "us-east-1": "US East (N. Virginia)",
    "us-east-2": "US East (Ohio)",
    "us-west-1": "US West (N. California)",
    "us-west-2": "US West (Oregon)",
    "ca-central-1": "Canada (Central)",
    "eu-west-1": "EU (Ireland)",
    "eu-west-2": "EU (London)",
    "eu-west-3": "EU (Paris)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-central-2": "EU (Zurich)",
    "eu-north-1": "EU (Stockholm)",
    "eu-south-1": "EU (Milan)",
    "eu-south-2": "EU (Spain)",
    "ap-south-1": "Asia Pacific (Mumbai)",
    "ap-south-2": "Asia Pacific (Hyderabad)",
    "ap-southeast-1": "Asia Pacific (Singapore)",
    "ap-southeast-2": "Asia Pacific (Sydney)",
    "ap-southeast-3": "Asia Pacific (Jakarta)",
    "ap-southeast-4": "Asia Pacific (Melbourne)",
    "ap-northeast-1": "Asia Pacific (Tokyo)",
    "ap-northeast-2": "Asia Pacific (Seoul)",
    "ap-northeast-3": "Asia Pacific (Osaka)",
    "ap-east-1": "Asia Pacific (Hong Kong)",
    "sa-east-1": "South America (São Paulo)",
    "me-south-1": "Middle East (Bahrain)",
    "me-central-1": "Middle East (UAE)",
    "af-south-1": "Africa (Cape Town)",
}

PRESETS = {
    "g5": ["c6i.xlarge", "g5.xlarge", "g5.12xlarge", "p4d.24xlarge"],
    "g4dn": ["c6i.xlarge", "g4dn.xlarge", "g4dn.12xlarge", "p3.8xlarge"],
}

def describe_instance_types(ec2, instance_types: List[str]) -> Dict[str, dict]:
    out = {}
    for i in range(0, len(instance_types), 20):
        chunk = instance_types[i:i+20]
        resp = ec2.describe_instance_types(InstanceTypes=chunk)
        for it in resp["InstanceTypes"]:
            name = it["InstanceType"]
            vcpus = it["VCpuInfo"]["DefaultVCpus"]
            mem_gb = it["MemoryInfo"]["SizeInMiB"] / 1024.0
            gpus = 0
            if "GpuInfo" in it and it["GpuInfo"].get("Gpus"):
                gpus = sum(g["Count"] for g in it["GpuInfo"]["Gpus"])
            out[name] = {"vcpu": vcpus, "mem_gb": mem_gb, "gpus": gpus}
    return out

def get_ondemand_price_per_hour(pricing, instance_type: str, location: str, operating_system="Linux") -> float:
    f = [
        {"Type": "TERM_MATCH", "Field": "ServiceCode", "Value": "AmazonEC2"},
        {"Type": "TERM_MATCH", "Field": "productFamily", "Value": "Compute Instance"},
        {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
        {"Type": "TERM_MATCH", "Field": "location", "Value": location},
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": operating_system},
        {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
        {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
    ]
    data = pricing.get_products(ServiceCode="AmazonEC2", Filters=f, MaxResults=100)
    for price_item in data.get("PriceList", []):
        product = json.loads(price_item)
        terms = product.get("terms", {}).get("OnDemand", {})
        for term in terms.values():
            price_dimensions = term.get("priceDimensions", {})
            for pd in price_dimensions.values():
                if pd.get("unit") == "Hrs":
                    price_str = pd["pricePerUnit"].get("USD")
                    if price_str is not None:
                        return float(price_str)
    # fallback
    f[-1]["Value"] = "AllocatedCapacity"
    data = pricing.get_products(ServiceCode="AmazonEC2", Filters=f, MaxResults=100)
    for price_item in data.get("PriceList", []):
        product = json.loads(price_item)
        terms = product.get("terms", {}).get("OnDemand", {})
        for term in terms.values():
            price_dimensions = term.get("priceDimensions", {})
            for pd in price_dimensions.values():
                if pd.get("unit") == "Hrs":
                    price_str = pd["pricePerUnit"].get("USD")
                    if price_str is not None:
                        return float(price_str)
    raise RuntimeError(f"On-Demand price not found for {instance_type} in {location}")

def fit_unit_costs(rows):
    A = np.array([[r[0], r[1], r[2]] for r in rows], dtype=float)
    y = np.array([r[3] for r in rows], dtype=float)
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"cpu_per_hr": w[0], "mem_gb_per_hr": w[1], "gpu_per_hr": w[2]}

def normalize_weights_to_one(weights, ref_caps):
    vcpu, mem_gb, gpus = ref_caps
    denom = weights["cpu_per_hr"]*vcpu + weights["mem_gb_per_hr"]*mem_gb + weights["gpu_per_hr"]*gpus
    if denom <= 0:
        return {"CPU": 0.0, "Mem_GB": 0.0, "GPU": 0.0}
    scale = 1.0 / denom
    return {
        "CPU": weights["cpu_per_hr"] * scale,
        "Mem_GB": weights["mem_gb_per_hr"] * scale,
        "GPU": weights["gpu_per_hr"] * scale,
    }

def main():
    ap = argparse.ArgumentParser(description="Derive TRES-like weights from EC2 specs and On-Demand prices.")
    ap.add_argument("--region", required=True, help="EC2 region code (e.g., eu-north-1)")
    ap.add_argument("--instances", nargs="+", help="Instance types to include")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), help="Use a predefined instance set")
    ap.add_argument("--ref", help="Reference instance type to normalize weights (defaults to first GPU instance)")
    ap.add_argument("--os", default="Linux", help="Operating system for pricing (default: Linux)")
    args = ap.parse_args()

    if not args.instances and not args.preset:
        print("Provide --instances or --preset. Example: --preset g5")
        sys.exit(1)

    instance_types = args.instances or PRESETS[args.preset]

    region = args.region
    location = REGION_TO_LOCATION.get(region)
    if not location:
        print(f"Region {region} not in map. Please add REGION_TO_LOCATION mapping for Pricing location string.")
        sys.exit(2)

    ec2 = boto3.client("ec2", region_name=region)
    pricing = boto3.client("pricing", region_name="us-east-1")

    specs = describe_instance_types(ec2, instance_types)
    rows = []
    print(f"\nCollecting specs & prices for location: {location}\n")
    for it in instance_types:
        if it not in specs:
            print(f"Skipping {it}: not returned by DescribeInstanceTypes", file=sys.stderr)
            continue
        vcpu = specs[it]["vcpu"]
        mem_gb = specs[it]["mem_gb"]
        gpus = specs[it]["gpus"]
        try:
            price = get_ondemand_price_per_hour(pricing, it, location, operating_system=args.os)
        except Exception as e:
            print(f"Error price {it}: {e}", file=sys.stderr)
            continue
        rows.append((vcpu, mem_gb, gpus, price))
        print(f"{it:15s}  vCPU={vcpu:3d}  MEM={mem_gb:6.1f} GB  GPU={gpus:1d}  Price=${price:.4f}/hr")

    if len(rows) < 2:
        print("\nNeed at least two instances (ideally 1 non-GPU + 1+ GPU) to fit costs.")
        sys.exit(3)

    unit = fit_unit_costs(rows)
    print("\nEstimated unit costs (per hour):")
    print(f"  CPU:     ${unit['cpu_per_hr']:.6f} per vCPU-hr")
    print(f"  Mem_GB:  ${unit['mem_gb_per_hr']:.6f} per GB-hr")
    print(f"  GPU:     ${unit['gpu_per_hr']:.6f} per GPU-hr")

    # Reference instance
    ref_type = args.ref
    if not ref_type:
        # choose first GPU instance if present
        ref_type = next((it for it in instance_types if specs[it]["gpus"] > 0), instance_types[0])
    ref_caps = (specs[ref_type]["vcpu"], specs[ref_type]["mem_gb"], specs[ref_type]["gpus"])

    w_norm = normalize_weights_to_one(unit, ref_caps)

    print(f"\nReference instance for normalization: {ref_type} "
          f"(vCPU={ref_caps[0]}, Mem_GB={ref_caps[1]:.1f}, GPU={ref_caps[2]})")
    print("\nSuggested TRESBillingWeights (normalized so a full node of reference ≈ 1.0):")
    print(f'  CPU={w_norm["CPU"]:.6f},Mem={w_norm["Mem_GB"]:.6f}G,GRES/gpu={w_norm["GPU"]:.6f}')

    cpu_base = unit["cpu_per_hr"] if unit["cpu_per_hr"] > 0 else 1.0
    rel_mem = unit["mem_gb_per_hr"] / cpu_base
    rel_gpu = unit["gpu_per_hr"] / cpu_base
    print("\nRelative costs (CPU = 1.0):")
    print(f"  Mem (per GB): {rel_mem:.3f} x CPU")
    print(f"  GPU (per GPU): {rel_gpu:.3f} x CPU")

    # Save JSON
    out = {
        "region": region,
        "location": location,
        "instances": [
            {"instance_type": it, "vcpu": specs[it]["vcpu"], "mem_gb": specs[it]["mem_gb"],
             "gpus": specs[it]["gpus"],
             "price_per_hour": next((r[3] for r in rows if r[0]==specs[it]['vcpu'] and r[1]==specs[it]['mem_gb'] and r[2]==specs[it]['gpus']), None)}
            for it in instance_types if it in specs
        ],
        "unit_costs_per_hour": unit,
        "tres_weights_normalized_ref": {
            "reference_instance": ref_type,
            "CPU": w_norm["CPU"],
            "Mem_GB": w_norm["Mem_GB"],
            "GPU": w_norm["GPU"],
            "slurm_conf": f'CPU={w_norm["CPU"]:.6f},Mem={w_norm["Mem_GB"]:.6f}G,GRES/gpu={w_norm["GPU"]:.6f}',
        },
        "relative_costs_cpu_base_1": {"Mem_GB": rel_mem, "GPU": rel_gpu},
    }
    with open("derived_tres_weights.json", "w") as f:
        json.dump(out, f, indent=2)
    print('\nSaved details to derived_tres_weights.json')

if __name__ == "__main__":
    try:
        main()
    except botocore.exceptions.NoCredentialsError:
        print("ERROR: No AWS credentials found. Run `aws configure`.", file=sys.stderr)
        sys.exit(4)
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(5)
