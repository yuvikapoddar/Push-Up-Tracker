import { useState } from "react";

function App() {
  const [report, setReport] = useState([]);
  const [videoUrl, setVideoUrl] = useState("");

  const handleUpload = async (e) => {
    const formData = new FormData();
    formData.append("file", e.target.files[0]);
  
    const res = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
  
    const json = await res.json();
  
    // update state
    setReport(json.rep_report || []);
    // 🔥 add timestamp query so browser reloads new video
    if (json.annotated_video) {
      setVideoUrl(json.annotated_video + "?t=" + new Date().getTime());
    }
  };
  

  return (
    <div style={{ maxWidth: 900, margin: "20px auto", fontFamily: "sans-serif" }}>
      <h2>Push-Up Counter</h2>
      <input type="file" accept="video/*" onChange={handleUpload} />
      
      {videoUrl && (
        <video
          src={videoUrl}
          controls
          style={{ width: "100%", marginTop: 20 }}
        />
      )}
      
      {report.length > 0 && (
        <table border="1" cellPadding="8" style={{ marginTop: 20, width: "100%" }}>
          <thead>
            <tr>
              <th>Rep</th>
              <th>ROM%</th>
              <th>Total Time (s)</th>
              <th>Eccentric (s)</th>
              <th>Concentric (s)</th>
            </tr>
          </thead>
          <tbody>
            {report.map((r) => (
              <tr key={r.rep_index}>
                <td>{r.rep_index}</td>
                <td>{r.rom_pct}</td>
                <td>{r.rep_time_s}</td>
                <td>{r.ecc_time_s}</td>
                <td>{r.con_time_s}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default App;
