import type { Metadata } from "next";
import "./globals.css";
import { ThemeProvider } from "./contexts/ThemeContext";
import { Header } from "./components/Header";

export const metadata: Metadata = {
  title: "DocFoundry - Intelligent Document Search",
  description: "Advanced document search and discovery platform with semantic understanding",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased bg-background text-foreground transition-colors duration-300">
        <ThemeProvider>
          <div className="min-h-screen bg-background">
            <Header />
            
            {/* Main content */}
            <main className="flex-1">
              {children}
            </main>
            
            {/* Footer */}
            <footer className="border-t border-border bg-background">
              <div className="container mx-auto px-4 py-6">
                <div className="flex flex-col sm:flex-row justify-between items-center text-sm text-muted-foreground">
                  <p>© 2024 DocFoundry. Advanced document search platform.</p>
                  <p className="mt-2 sm:mt-0">
                    Powered by semantic understanding and AI
                  </p>
                </div>
              </div>
            </footer>
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
