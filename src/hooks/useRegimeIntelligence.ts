import { useEffect, useState } from "react";

export function useRegimeIntelligence() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetch("/api/regime-intelligence")
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);

  return data;
}