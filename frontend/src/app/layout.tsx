import "./globals.css";
import type { Metadata } from "next";
import { Shell } from "@/components/Shell";

export const metadata: Metadata = {
  title: "Factor Terminal",
  description:
    "Daily return-based multi-asset factor model: forty factors across nine blocks",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg font-sans antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
