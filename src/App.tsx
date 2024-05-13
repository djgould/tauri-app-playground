import { useState } from "react";
import reactLogo from "./assets/react.svg";
import { invoke } from "@tauri-apps/api/tauri";
import { Button } from "./components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/ui/table";

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [greetMsg, setGreetMsg] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  async function greet() {
    setLoading(true);
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    try {
      const value = (await invoke("transcribe", {
        path: "./samples/a13.wav",
      })) as string[];
      setGreetMsg(value);
    } finally {
      setLoading(false);
    }
  }

  async function recordTenSeconds() {
    await invoke("record");
  }

  async function startRecording() {
    setIsRecording(true);
    await invoke("start_recording").catch((err) => {
      console.error(err);
      setIsRecording(false);
    });
  }

  async function stopRecording() {
    setIsRecording(false);
    await invoke("stop_recording").catch((err) => {
      console.error(err);
    });
  }

  return (
    <div className="flex flex-col max-h-screen">
      <header className="h-16 flex justify-center p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            greet();
          }}
        >
          <Button disabled={loading} type="submit">
            {loading ? "Loading..." : "Transcribe"}
          </Button>
        </form>
      </header>
      <Table className="flex-grow flex-shrink overflow-y-scroll h-40">
        <TableHead>
          <TableRow>
            <TableHeader>Speaker</TableHeader>
            <TableHeader>Transcription</TableHeader>
          </TableRow>
        </TableHead>
        <TableBody>
          {greetMsg.map((msg, i) => (
            <TableRow key={i}>
              <TableCell className="font-medium">
                {i > 0 ? "<SPEAKER NEXT>" : "<SPEAKER 1>"}
              </TableCell>
              <TableCell>{msg}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <footer className="h-16 flex justify-center gap-4 p-4">
        <Button onClick={() => recordTenSeconds()}>Record 10 Seconds</Button>
        <Button
          onClick={() => (isRecording ? stopRecording() : startRecording())}
        >
          {isRecording ? "Stop Recording" : "Start Recording"}
        </Button>
        <Button onClick={() => setGreetMsg([])}>Clear</Button>
      </footer>
    </div>
  );
}

export default App;
